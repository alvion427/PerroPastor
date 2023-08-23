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

  public static readonly Dictionary<QuantizationModes, string> QuantizationFlags =
    new Dictionary<QuantizationModes, string>() {
      { QuantizationModes.Float32, "QUANT_WEIGHT_32" },
      { QuantizationModes.Float16, "QUANT_WEIGHT_16" },
    };

  public static void EnableQuantizationKeywords(ComputeShader shader, QuantizationModes mode) {
    foreach (var item in QuantizationFlags) {
      if (item.Key == mode) {
        shader.EnableKeyword(item.Value);
      }
      else {
        shader.DisableKeyword(item.Value);
      }
    }
  }
}

public class LlamaConfig {
  public LlamaConfig(QuantizationModes quantizationMode) {
    quantization_mode = quantizationMode;
  }

  public readonly QuantizationModes quantization_mode;

  public int dim; // Transformer dimension
  public int hidden_dim; // For FFN layers
  public int n_layers; // Number of layers
  public int n_heads; // Number of query heads
  public int n_kv_heads; // Unused
  public int vocab_size; // Vocabulary size, usually 256 (byte-level)
  public int seq_len; // Max sequence length

  public int head_size => dim / n_heads;
  public bool UseSharedVocab => vocab_size > 0;

  public int QuantizedSize => QuantizationUtil.QuantizationSizes[quantization_mode];
  public int QuantizedSizeVec => 4 * QuantizedSize;
  public string QuantizationFlag => QuantizationUtil.QuantizationFlags[quantization_mode];
}

public class LayerWeights : IDisposable {
  public NativeArray<float> rms_att_weight; // (dim) RMSNorm weights
  public NativeArray<float> rms_ffn_weight; // (dim)

  public NativeArray<float> wq; // (dim, dim)
  public NativeArray<float> wk; // (dim, dim)
  public NativeArray<float> wv; // (dim, dim)
  public NativeArray<float> wo; // (dim, dim)

  public NativeArray<float> w1; // (hidden_dim, dim)
  public NativeArray<float> w2; // (dim, hidden_dim)
  public NativeArray<float> w3; // (hidden_dim, dim)

  public LayerWeights(LlamaConfig c) {
    rms_att_weight = new NativeArray<float>(c.dim, Allocator.Persistent);
    rms_ffn_weight = new NativeArray<float>(c.dim, Allocator.Persistent);

    wq = new NativeArray<float>(c.dim * c.dim, Allocator.Persistent);
    wk = new NativeArray<float>(c.dim * c.dim, Allocator.Persistent);
    wv = new NativeArray<float>(c.dim * c.dim, Allocator.Persistent);
    wo = new NativeArray<float>(c.dim * c.dim, Allocator.Persistent);

    w1 = new NativeArray<float>(c.hidden_dim * c.dim, Allocator.Persistent);
    w2 = new NativeArray<float>(c.dim * c.hidden_dim, Allocator.Persistent);
    w3 = new NativeArray<float>(c.hidden_dim * c.dim, Allocator.Persistent);
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
  public NativeArray<float> token_embedding_table; // (vocab_size, dim)

  public NativeArray<float> rms_final_weight; // (dim) RMSNorm weights
  public NativeArray<float> freq_cis_real; // (seq_len, head_size/2)
  public NativeArray<float> freq_cis_imag; // (seq_len, head_size/2)

  public NativeArray<float> wcls; // (vocab_size, dim)

  public LayerWeights[] layerWeights;

  public Weights(LlamaConfig c) {
    int headSize = c.dim / c.n_heads;

    token_embedding_table = new NativeArray<float>(c.vocab_size * c.dim, Allocator.Persistent);
    rms_final_weight = new NativeArray<float>(c.dim, Allocator.Persistent);
    freq_cis_real = new NativeArray<float>(c.seq_len * headSize / 2, Allocator.Persistent);
    freq_cis_imag = new NativeArray<float>(c.seq_len * headSize / 2, Allocator.Persistent);
    if (!c.UseSharedVocab) {
      wcls = new NativeArray<float>(c.vocab_size * c.dim, Allocator.Persistent);
    }

    layerWeights = new LayerWeights[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layerWeights[layer] = new LayerWeights(c);
    }
  }

  public void Dispose() {
    token_embedding_table.Dispose();
    rms_final_weight.Dispose();
    freq_cis_real.Dispose();
    freq_cis_imag.Dispose();
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
  public ComputeBuffer freq_cis;
  public ComputeBuffer wcls;

  public LayerWeightsGPU[] layerWeights;

  public ComputeBuffer GetWCLS() => wcls ?? token_embedding_table;

  public WeightsGPU(LlamaConfig c) {
    token_embedding_table = CreateWeightsBuffer(c, c.vocab_size * c.dim);
    rms_final_weight = CreateWeightsBuffer(c, c.dim);
    freq_cis = CreateWeightsBuffer(c, c.seq_len * c.head_size);
    if (!c.UseSharedVocab)
      wcls = CreateWeightsBuffer(c, c.vocab_size * c.dim);

    layerWeights = new LayerWeightsGPU[c.n_layers];
    for (int layer = 0; layer < c.n_layers; layer++) {
      layerWeights[layer] = new LayerWeightsGPU(c);
    }
  }

  public void Dispose() {
    token_embedding_table.Dispose();
    rms_final_weight.Dispose();
    freq_cis.Dispose();
    if (wcls != null)
      wcls.Dispose();

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].Dispose();
    }
  }

  public void LoadWeights(LlamaConfig c, Weights weights) {
    ComputeUtils.SetQuantizedData(c.quantization_mode, token_embedding_table, weights.token_embedding_table);
    ComputeUtils.SetQuantizedData(c.quantization_mode, rms_final_weight, weights.rms_final_weight);
    ComputeUtils.SetQuantizedDataInterleaved(c.quantization_mode, freq_cis, weights.freq_cis_real,
      weights.freq_cis_imag);

    if (wcls != null)
      ComputeUtils.SetQuantizedData(c.quantization_mode, wcls, weights.wcls);

    for (int layer = 0; layer < layerWeights.Length; layer++) {
      layerWeights[layer].LoadWeights(c, weights.layerWeights[layer]);
    }
  }

  public static ComputeBuffer CreateWeightsBuffer(LlamaConfig config, int size) {
    return ComputeUtils.CreateVectorizedBuffer(size, config.QuantizedSize);
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

  public void LoadWeights(LlamaConfig c, LayerWeights weights) {
    ComputeUtils.SetQuantizedData(c.quantization_mode, rms_att_weight, weights.rms_att_weight);
    ComputeUtils.SetQuantizedData(c.quantization_mode, rms_ffn_weight, weights.rms_ffn_weight);

    ComputeUtils.SetQuantizedData(c.quantization_mode, wq, weights.wq);
    ComputeUtils.SetQuantizedData(c.quantization_mode, wk, weights.wk);
    ComputeUtils.SetQuantizedData(c.quantization_mode, wv, weights.wv);
    ComputeUtils.SetQuantizedData(c.quantization_mode, wo, weights.wo);

    ComputeUtils.SetQuantizedData(c.quantization_mode, w1, weights.w1);
    ComputeUtils.SetQuantizedData(c.quantization_mode, w2, weights.w2);
    ComputeUtils.SetQuantizedData(c.quantization_mode, w3, weights.w3);
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
    key_cache = new ComputeBuffer(c.seq_len * c.dim, sizeof(float));
    value_cache = new ComputeBuffer(c.seq_len * c.dim, sizeof(float));
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
  public int softmax;
  public int silu;
  public int multiply;
  public int weightedSum;
  public int sampleLogits;
  public int findMaxIdx;

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
    softmax = shader.FindKernel("Softmax");
    silu = shader.FindKernel("Silu");
    multiply = shader.FindKernel("Multiply");
    weightedSum = shader.FindKernel("WeightedSum");
    sampleLogits = shader.FindKernel("SampleLogits");
    findMaxIdx = shader.FindKernel("FindMaxIndex");
  }
}