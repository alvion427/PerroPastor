using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling.Memory.Experimental;

[Serializable]
public class LlamaConfig {
  public int dim; // Transformer dimension
  public int hidden_dim; // For FFN layers
  public int n_layers; // Number of layers
  public int n_heads; // Number of query heads
  public int n_kv_heads; // Unused
  public int vocab_size; // Vocabulary size, usually 256 (byte-level)
  public int seq_len; // Max sequence length

  public int head_size => dim / n_heads;

  public override int GetHashCode()
  {
    return dim.GetHashCode() ^ 
           hidden_dim.GetHashCode() ^ 
           n_layers.GetHashCode() ^ 
           n_heads.GetHashCode() ^ 
           n_kv_heads.GetHashCode() ^ 
           vocab_size.GetHashCode() ^ 
           seq_len.GetHashCode();
  }
  
  public override bool Equals(object obj) {
    if (obj is LlamaConfig other) {
      return dim == other.dim &&
             hidden_dim == other.hidden_dim &&
             n_layers == other.n_layers &&
             n_heads == other.n_heads &&
             n_kv_heads == other.n_kv_heads &&
             vocab_size == other.vocab_size &&
             seq_len == other.seq_len;
    }

    return false;
  }
}

public class GpuTensor : IDisposable {
  public Vector2Int Shape;
  public QuantizationModes Mode;
  public ComputeBuffer Buffer;

  public long Size => Shape.x * Shape.y;
  public long SizeBytes => Buffer.count * BlockSizeBytes;
  public int BlockCount => (int)(Size / BlockSize);
  public int BlockSize => Mode.BlockSize();
  public int BlockSizeBytes => Mode.BlockSizeBytes();
  
  public GpuTensor(int rows, int cols, QuantizationModes mode) {
    Shape = new Vector2Int(rows, cols);
    Mode = mode;
    Buffer = ComputeUtils.CreateBlockBuffer(rows * cols, mode);
  }

  public void Dispose() {
    Buffer?.Dispose();
  }
}

public class WeightsGpu {
  public GpuTensor token_embedding_table;
  public GpuTensor rms_final_weight;
  public GpuTensor wcls;

  public LayerWeightsGPU[] layerWeights;

  public GpuTensor GetWCLS() => wcls ?? token_embedding_table;
  
  public int ReferenceCount { get; private set; }

  public WeightsGpu() {
    ReferenceCount = 1;
  }

  public void AddReference() {
    if (ReferenceCount == 0) {
      throw new Exception("Cannot add reference to disposed weights");
    }
    ReferenceCount++;
  }
  
  public void RemoveReference() {
    if (ReferenceCount == 0) {
      throw new Exception("Cannot remove reference from disposed weights");
    }
    ReferenceCount--;
    if (ReferenceCount == 0) {
      Dispose();
    }
  }

  public bool IsValid() {
    if (ReferenceCount > 0) {
      if (!token_embedding_table.Buffer.IsValid() || !rms_final_weight.Buffer.IsValid() || (wcls != null && !wcls.Buffer.IsValid())) {
        return false;
      }
      
      for (int l = 0; l < layerWeights.Length; l++) {
        if (!layerWeights[l].IsValid()) {
          return false;
        }
      }
      
      return true;
    }
    else {
      return false;
    }
  }

  private void Dispose() {
    token_embedding_table.Dispose();
    rms_final_weight.Dispose();
    if (wcls != null)
      wcls.Dispose();

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].Dispose();
    }
  }
}

public class LayerWeightsGPU : IDisposable {
  public GpuTensor rms_att_weight;
  public GpuTensor rms_ffn_weight;

  public GpuTensor wq;
  public GpuTensor wk;
  public GpuTensor wv;
  public GpuTensor wo;

  public GpuTensor w1;
  public GpuTensor w2;
  public GpuTensor w3;

  public void Dispose() {
    rms_att_weight.Dispose();
    rms_ffn_weight.Dispose();

    wq.Dispose();
    wk.Dispose();
    wv.Dispose();
    wo.Dispose();

    w1.Dispose();
    w2.Dispose();
    w3.Dispose();
  }

  public bool IsValid() {
    return rms_att_weight.Buffer.IsValid() && rms_ffn_weight.Buffer.IsValid() &&
           wq.Buffer.IsValid() && wk.Buffer.IsValid() && wv.Buffer.IsValid() && wo.Buffer.IsValid() &&
           w1.Buffer.IsValid() && w2.Buffer.IsValid() && w3.Buffer.IsValid();
  }
}

public class LayerPersistentState : IDisposable {
  public ComputeBuffer key_cache; // (seq_len, dim)
  public ComputeBuffer value_cache; // (seq_len, dim)

  public LayerPersistentState(LlamaConfig c, QuantizationModes quantMode) {
    key_cache = ComputeUtils.CreateBlockBuffer(c.seq_len * c.dim, quantMode);
    value_cache = ComputeUtils.CreateBlockBuffer(c.seq_len * c.dim, quantMode);
  }

  public void Dispose() {
    key_cache.Dispose();
    value_cache.Dispose();
  }
}

// The only side effects we care about by running the transformer are the token that is output
// and the mutations to this state.
public class PersistentState : IDisposable {
  public LayerPersistentState[] layers;

  public PersistentState(LlamaConfig c, QuantizationModes quantMode) {
    layers = new LayerPersistentState[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layers[layer] = new LayerPersistentState(c, quantMode);
    }
  }

  public void Dispose() {
    for (int layer = 0; layer < layers.Length; layer++) {
      layers[layer].Dispose();
    }
  }
}

// Compute buffers for running state.  All of these are temporary and thrown away at the end of a layer. 
public class RunState : IDisposable {
  // Serves as the input into each layer.  For layer 0, it's the embedding of the last token.
  // For subsequent layers, it's the output of the previous layer.
  public ComputeBuffer x;
  public ComputeBuffer xb; // same, but inside a residual branch (dim,)
  public ComputeBuffer xb2; // an additional buffer just for convenience (dim,)
  public ComputeBuffer xbFixed; // used to output (dim,) vectors as fixed point
  public ComputeBuffer hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  public ComputeBuffer hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  public ComputeBuffer hbFixed; // used to output (hidden_dim,) vectors as fixed point
  public ComputeBuffer q; // query (dim,)
  public ComputeBuffer k; // key (dim,)
  public ComputeBuffer v; // value (dim,)
  public ComputeBuffer att; // buffer for scores/attention values (n_heads, seq_len)
  public ComputeBuffer logits; // output logits
  public ComputeBuffer logitsFixed;

  // Due to a very bad implementation of softmax we need to use a temporary buffer, but we can remove this once we
  // improve it.
  public ComputeBuffer softmaxTemp;
  public ComputeBuffer softmaxTempB;

  public ComputeBuffer scalarTemp0;
  public ComputeBuffer scalarTemp1;

  public RunState(LlamaConfig c, QuantizationModes rutnimeQuantizationMode) {
    x = new ComputeBuffer(c.dim, sizeof(float));
    xb = new ComputeBuffer(c.dim, sizeof(float));
    xb2 = new ComputeBuffer(c.dim, sizeof(float));
    xbFixed = new ComputeBuffer(c.dim, sizeof(int));
    hb = new ComputeBuffer(c.hidden_dim, sizeof(float));
    hb2 = new ComputeBuffer(c.hidden_dim, sizeof(float));
    hbFixed = new ComputeBuffer(c.hidden_dim, sizeof(int));
    q = new ComputeBuffer(c.dim, sizeof(float));
    k = new ComputeBuffer(c.dim, sizeof(float));
    v = new ComputeBuffer(c.dim, sizeof(float));
    att = new ComputeBuffer(c.n_heads * c.seq_len, sizeof(float));
    logits = new ComputeBuffer(c.vocab_size, sizeof(float));
    logitsFixed = new ComputeBuffer(c.vocab_size, sizeof(int));

    softmaxTemp = new ComputeBuffer(c.vocab_size, sizeof(float));
    softmaxTempB = new ComputeBuffer(c.vocab_size, sizeof(float));
    
    scalarTemp0 = new ComputeBuffer(1, sizeof(float));
    scalarTemp1 = new ComputeBuffer(1, sizeof(float));
  }

  public void Dispose() {
    x.Dispose();
    xb.Dispose();
    xb2.Dispose();
    xbFixed.Dispose();
    hb.Dispose();
    hb2.Dispose();
    hbFixed.Dispose();
    q.Dispose();
    k.Dispose();
    v.Dispose();
    att.Dispose();
    logits.Dispose();
    logitsFixed.Dispose();
    softmaxTemp.Dispose();
    softmaxTempB.Dispose();
    scalarTemp0.Dispose();
    scalarTemp1.Dispose();
  }
}

public class LlamaKernels {
  public int clear;
  public int memcpy;
  public int scaleBuffer;
  public int fixedToFloat;
  public int loadEmbedding;
  public int matmul;
  public int accumulate;
  public int rmsNorm;
  public int rope;
  public int computeAttention;
  public int softmaxExp;
  public int softmaxDivide;
  public int silu;
  public int multiply;
  public int weightedSum;
  public int sampleLogits;
  public int findMaxIdx;
  public int findMaxVal;

  public LlamaKernels(ComputeShader shader) {
    clear = shader.FindKernel("Clear");
    memcpy = shader.FindKernel("Memcpy");
    scaleBuffer = shader.FindKernel("ScaleBuffer");
    loadEmbedding = shader.FindKernel("LoadEmbedding");
    fixedToFloat = shader.FindKernel("FixedToFloat");
    matmul = shader.FindKernel("MatMul");
    accumulate = shader.FindKernel("Accumulate");
    rmsNorm = shader.FindKernel("RMSNorm");
    rope = shader.FindKernel("Rope");
    computeAttention = shader.FindKernel("ComputeAttention");
    softmaxExp = shader.FindKernel("SoftmaxExp");
    softmaxDivide = shader.FindKernel("SoftmaxDivide");
    silu = shader.FindKernel("Silu");
    multiply = shader.FindKernel("Multiply");
    weightedSum = shader.FindKernel("WeightedSum");
    sampleLogits = shader.FindKernel("SampleLogits");
    findMaxIdx = shader.FindKernel("FindMaxIndex");
    findMaxVal = shader.FindKernel("FindMaxValue");
  }
}