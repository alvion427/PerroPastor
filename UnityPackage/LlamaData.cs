using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling.Memory.Experimental;

public enum QuantizationModes {
  // unquantized
  Float32,
  Float16,
  
  // ggml types
  Q8_0,   
}

public static class QuantizationUtil {
  public static int BlockSize(this QuantizationModes quantMode) {
    return BlockSizes[quantMode];
  }

  public static int BlockSizeBytes(this QuantizationModes quantMode) {
    return BlockByteSizes[quantMode];
  }

  public static void EnableQuantizationKeywords(ComputeShader shader, QuantizationModes mode, string prefix) {
    foreach (var item in QuantizationFlags) {
      string keyword = string.Format(item.Value, prefix);
      if (item.Key == mode) {
        shader.EnableKeyword(keyword);
      }
      else {
        shader.DisableKeyword(keyword);
      }
    }
  }

  public static int ElementSize(this QuantizationModes quantMode) {
    if (quantMode == QuantizationModes.Float32) {
      return 4;
    }
    else if (quantMode == QuantizationModes.Float16) {
      return 2;
    }
    else {
      throw new ArgumentException("Quantized buffers don't have an 'ElementSize'");
    }
  }
  
  private static readonly Dictionary<QuantizationModes, string> QuantizationFlags =
    new Dictionary<QuantizationModes, string>() {
      { QuantizationModes.Float32, "QUANT_{0}_32" },
      { QuantizationModes.Float16, "QUANT_{0}_16" },
      { QuantizationModes.Q8_0, "QUANT_{0}_Q8_0"}
    };
  
  private static readonly Dictionary<QuantizationModes, int> BlockSizes = 
    new Dictionary<QuantizationModes, int>() {
      { QuantizationModes.Float32, 4 },
      { QuantizationModes.Float16, 4 },
      { QuantizationModes.Q8_0, 32}
    };

  private static readonly Dictionary<QuantizationModes, int> BlockByteSizes = 
    new Dictionary<QuantizationModes, int>() {
      { QuantizationModes.Float32, 16 },
      { QuantizationModes.Float16, 8 },
      { QuantizationModes.Q8_0, 36}
    };
  
  public static unsafe float[] DequantizeCpu(ComputeBuffer quantizedBuffer, QuantizationModes mode, 
    int offset = 0, int length = 0) {
    int numBlocks = quantizedBuffer.count;
    int blockSize = mode.BlockSize();
    int elementCount = numBlocks * blockSize;

    length = length == 0 ? numBlocks - offset : length; 

    float[] result = new float[elementCount];
    if (mode == QuantizationModes.Q8_0) {
      Q8_0Block[] quantizedData = new Q8_0Block[quantizedBuffer.count];
      quantizedBuffer.GetData(quantizedData);

      if (offset + length > elementCount) {
        throw new ArgumentException($"Not enough space in buffer: {offset} + {length} > {elementCount}");
      }
      for (int b = offset; b < offset + length; ++b) {
        float scale = quantizedData[b].scale;
        for (int i = 0; i < blockSize; ++i) {
          int idx = (b - offset) * blockSize + i;
          result[idx] = quantizedData[b].values[i] * scale;
        }
      }
    }
    else {
      throw new ArgumentException("No dequantize implementation for " + mode);
    }

    return result;
  }
}

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

public class WeightsGpu : IDisposable {
  public GpuTensor token_embedding_table;
  public GpuTensor rms_final_weight;
  public GpuTensor wcls;

  public LayerWeightsGPU[] layerWeights;

  public GpuTensor GetWCLS() => wcls ?? token_embedding_table;

  public void Dispose() {
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
  public ComputeBuffer xbFixed; // used to output weighted sum as fixed point to use atomics
  public ComputeBuffer hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  public ComputeBuffer hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  public ComputeBuffer q; // query (dim,)
  public ComputeBuffer k; // key (dim,)
  public ComputeBuffer v; // value (dim,)
  public ComputeBuffer att; // buffer for scores/attention values (n_heads, seq_len)
  public ComputeBuffer logits; // output logits
  public ComputeBuffer outputToken; // output token

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
    q = new ComputeBuffer(c.dim, sizeof(float));
    k = new ComputeBuffer(c.dim, sizeof(float));
    v = new ComputeBuffer(c.dim, sizeof(float));
    att = new ComputeBuffer(c.n_heads * c.seq_len, sizeof(float));
    logits = new ComputeBuffer(c.vocab_size, sizeof(float));
    outputToken = new ComputeBuffer(1, sizeof(int));

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
    q.Dispose();
    k.Dispose();
    v.Dispose();
    att.Dispose();
    logits.Dispose();
    outputToken.Dispose();
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