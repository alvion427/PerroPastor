using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

public enum QuantizationModes {
  Float32,
  Float16,
}

public static class QuantizationUtil {
  public static int ElementSize(this QuantizationModes mode) {
    return QuantizationSizes[mode];
  }
  
  public static readonly Dictionary<QuantizationModes, RenderTextureFormat> QuantizationFormats =
    new Dictionary<QuantizationModes, RenderTextureFormat>() {
      { QuantizationModes.Float32, RenderTextureFormat.RFloat },
      { QuantizationModes.Float16, RenderTextureFormat.RHalf }
    };

  public static readonly Dictionary<QuantizationModes, int> QuantizationSizes =
    new Dictionary<QuantizationModes, int>() {
      { QuantizationModes.Float32, Marshal.SizeOf<float>() },
      { QuantizationModes.Float16, Marshal.SizeOf<half>() },
    };

  private static readonly Dictionary<QuantizationModes, string> QuantizationFlags =
    new Dictionary<QuantizationModes, string>() {
      { QuantizationModes.Float32, "QUANT_{0}_32" },
      { QuantizationModes.Float16, "QUANT_{0}_16" },
    };


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
}

[Serializable]
public class LlamaConfig {
  public LlamaConfig(
    QuantizationModes sourceQuantizationMode, 
    QuantizationModes weightQuantizationMode, 
    QuantizationModes runtimeQuantizationMode) {

    source_quantization_mode = sourceQuantizationMode;
    weight_quantization_mode = weightQuantizationMode;
    runtime_quantization_mode = runtimeQuantizationMode;
  }

  public readonly QuantizationModes source_quantization_mode;
  public readonly QuantizationModes weight_quantization_mode;
  public readonly QuantizationModes runtime_quantization_mode;

  public int dim; // Transformer dimension
  public int hidden_dim; // For FFN layers
  public int n_layers; // Number of layers
  public int n_heads; // Number of query heads
  public int n_kv_heads; // Unused
  public int vocab_size; // Vocabulary size, usually 256 (byte-level)
  public int seq_len; // Max sequence length

  public int head_size => dim / n_heads;
}

public class LayerWeights : IDisposable {
  public NativeArray<byte> rms_att_weight; // (dim) RMSNorm weights
  public NativeArray<byte> rms_ffn_weight; // (dim)

  public NativeArray<byte> wq; // (dim, dim)
  public NativeArray<byte> wk; // (dim, dim)
  public NativeArray<byte> wv; // (dim, dim)
  public NativeArray<byte> wo; // (dim, dim)

  public NativeArray<byte> w1; // (hidden_dim, dim)
  public NativeArray<byte> w2; // (dim, hidden_dim)
  public NativeArray<byte> w3; // (hidden_dim, dim)

  public LayerWeights(LlamaConfig c, QuantizationModes quantMode) {
    int weightSize = quantMode.ElementSize();
    rms_att_weight = new NativeArray<byte>(c.dim * weightSize, Allocator.Persistent);
    rms_ffn_weight = new NativeArray<byte>(c.dim * weightSize, Allocator.Persistent);

    wq = new NativeArray<byte>(c.dim * c.dim * weightSize, Allocator.Persistent);
    wk = new NativeArray<byte>(c.dim * c.dim * weightSize, Allocator.Persistent);
    wv = new NativeArray<byte>(c.dim * c.dim * weightSize, Allocator.Persistent);
    wo = new NativeArray<byte>(c.dim * c.dim * weightSize, Allocator.Persistent);

    w1 = new NativeArray<byte>(c.hidden_dim * c.dim * weightSize, Allocator.Persistent);
    w2 = new NativeArray<byte>(c.dim * c.hidden_dim * weightSize, Allocator.Persistent);
    w3 = new NativeArray<byte>(c.hidden_dim * c.dim * weightSize, Allocator.Persistent);
  }

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

public class Weights : IDisposable {
  public QuantizationModes QuantizationMode { get; private set; }
  public int WeightSize => QuantizationMode.ElementSize();
  public bool HasClassifierWeights { get => wcls.IsCreated; }
  
  public NativeArray<byte> token_embedding_table; // (vocab_size, dim)
  public NativeArray<byte> rms_final_weight; // (dim) RMSNorm weights
  public NativeArray<byte> wcls; // (vocab_size, dim)

  public LayerWeights[] layerWeights;

  public Weights(LlamaConfig c, QuantizationModes quantMode, bool hasClassifierWeights) {
    QuantizationMode = quantMode;
    
    int headSize = c.dim / c.n_heads;

    token_embedding_table = new NativeArray<byte>(c.vocab_size * c.dim * WeightSize, Allocator.Persistent);
    rms_final_weight = new NativeArray<byte>(c.dim * WeightSize, Allocator.Persistent);
    if (hasClassifierWeights) {
      wcls = new NativeArray<byte>(c.vocab_size * c.dim * WeightSize, Allocator.Persistent);
    }

    layerWeights = new LayerWeights[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layerWeights[layer] = new LayerWeights(c, quantMode);
    }
  }

  public void Dispose() {
    token_embedding_table.Dispose();
    rms_final_weight.Dispose();
    if (wcls.IsCreated) {
      wcls.Dispose();
    }

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].Dispose();
    }
  }
}

public class WeightsGPU : IDisposable {
  public ComputeBuffer token_embedding_table;
  public ComputeBuffer rms_final_weight;
  public ComputeBuffer wcls;

  public LayerWeightsGPU[] layerWeights;

  public ComputeBuffer GetWCLS() => wcls ?? token_embedding_table;

  public WeightsGPU(LlamaConfig c, bool hasClassifierWeights) {
    token_embedding_table = CreateWeightsBuffer(c, c.vocab_size * c.dim);
    rms_final_weight = CreateWeightsBuffer(c, c.dim);
    if (hasClassifierWeights)
      wcls = CreateWeightsBuffer(c, c.vocab_size * c.dim);

    layerWeights = new LayerWeightsGPU[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layerWeights[layer] = new LayerWeightsGPU(c);
    }
  }

  public void Dispose() {
    token_embedding_table.Dispose();
    rms_final_weight.Dispose();
    if (wcls != null)
      wcls.Dispose();

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].Dispose();
    }
  }

  public void LoadWeights(LlamaConfig c, Weights weights) {
    QuantizationModes sourceMode = weights.QuantizationMode;
    QuantizationModes destMode = c.weight_quantization_mode;
    ComputeUtils.SetQuantizedData(sourceMode, destMode, token_embedding_table, weights.token_embedding_table);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, rms_final_weight, weights.rms_final_weight);

    if (wcls != null)
      ComputeUtils.SetQuantizedData(sourceMode, destMode, wcls, weights.wcls);

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].LoadWeights(c, weights.layerWeights[layer], sourceMode);
    }
  }

  public static ComputeBuffer CreateWeightsBuffer(LlamaConfig config, int size) {
    return ComputeUtils.CreateVectorizedBuffer(size, config.weight_quantization_mode.ElementSize());
  }
}

public class LayerWeightsGPU : IDisposable {
  public ComputeBuffer rms_att_weight;
  public ComputeBuffer rms_ffn_weight;

  public ComputeBuffer wq;
  public ComputeBuffer wk;
  public ComputeBuffer wv;
  public ComputeBuffer wo;

  public ComputeBuffer w1;
  public ComputeBuffer w2;
  public ComputeBuffer w3;

  public LayerWeightsGPU(LlamaConfig c) {
    rms_att_weight = WeightsGPU.CreateWeightsBuffer(c, c.dim);
    rms_ffn_weight = WeightsGPU.CreateWeightsBuffer(c, c.dim);

    wq = WeightsGPU.CreateWeightsBuffer(c, c.dim * c.dim);
    wk = WeightsGPU.CreateWeightsBuffer(c, c.dim * c.dim);
    wv = WeightsGPU.CreateWeightsBuffer(c, c.dim * c.dim);
    wo = WeightsGPU.CreateWeightsBuffer(c, c.dim * c.dim);

    w1 = WeightsGPU.CreateWeightsBuffer(c, c.hidden_dim * c.dim);
    w2 = WeightsGPU.CreateWeightsBuffer(c, c.dim * c.hidden_dim);
    w3 = WeightsGPU.CreateWeightsBuffer(c, c.hidden_dim * c.dim);
  }

  public void LoadWeights(LlamaConfig c, LayerWeights weights, QuantizationModes sourceMode) {
    QuantizationModes destMode = c.weight_quantization_mode;
    
    ComputeUtils.SetQuantizedData(sourceMode, destMode, rms_att_weight, weights.rms_att_weight);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, rms_ffn_weight, weights.rms_ffn_weight);

    ComputeUtils.SetQuantizedData(sourceMode, destMode, wq, weights.wq);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, wk, weights.wk);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, wv, weights.wv);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, wo, weights.wo);

    ComputeUtils.SetQuantizedData(sourceMode, destMode, w1, weights.w1);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, w2, weights.w2);
    ComputeUtils.SetQuantizedData(sourceMode, destMode, w3, weights.w3);
  }

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

  public LayerPersistentState(LlamaConfig c) {
    int runtimeSize = c.runtime_quantization_mode.ElementSize();
    key_cache = ComputeUtils.CreateVectorizedBuffer(c.seq_len * c.dim, runtimeSize);
    value_cache = ComputeUtils.CreateVectorizedBuffer(c.seq_len * c.dim, runtimeSize);
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

  public PersistentState(LlamaConfig c) {
    layers = new LayerPersistentState[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layers[layer] = new LayerPersistentState(c);
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

  public RunState(LlamaConfig c) {
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
  public int matmulTex;
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
    matmulTex = shader.FindKernel("MatMulTex");
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