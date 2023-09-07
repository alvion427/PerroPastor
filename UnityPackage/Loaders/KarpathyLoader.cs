using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Profiling;

public class KarpathyLoader : ModelLoaderBase {
  public string VocabPath;
  public LlamaConfig Config;
  public QuantizationModes QuantMode;

  // We can eventually use fences to dump this intelligently, instead we just clear it out when we're done.
  private List<IDisposable> _garbage = new List<IDisposable>();

  protected async override Task<(LlamaConfig, WeightsGpu, Tokenizer)> LoadModelImpl() {
    string fullPath = GetFullModelPath();

    // Karpathy's format only uses float32
    float startTime = Time.realtimeSinceStartup;
    Debug.Log("Loading weights...");
    try {
      Profiler.BeginSample("LoadWeights");
      LlamaConfig config = null;
      WeightsGpu weights = null;

      using (FileStream fs = new FileStream(fullPath, FileMode.Open, FileAccess.Read))
      using (BinaryReader br = new BinaryReader(fs)) {
        // Read the config
        config = new LlamaConfig() {
          dim = br.ReadInt32(),
          hidden_dim = br.ReadInt32(),
          n_layers = br.ReadInt32(),
          n_heads = br.ReadInt32(),
          n_kv_heads = br.ReadInt32(),
          vocab_size = br.ReadInt32(),
          seq_len = br.ReadInt32()
        };

        config.vocab_size = Math.Abs(config.vocab_size);

        bool hasClassifierWeights = false;

        // Initialize weights
        weights = new WeightsGpu() {
          token_embedding_table = new GpuTensor(config.vocab_size, config.dim, QuantMode),
          rms_final_weight = new GpuTensor(config.dim, 1, QuantMode),
          layerWeights = new LayerWeightsGPU[config.n_layers],
        };

        for (int layer = 0; layer < config.n_layers; layer++) {
          weights.layerWeights[layer] = new LayerWeightsGPU() {
            rms_att_weight = new GpuTensor(config.dim, 1, QuantMode),
            rms_ffn_weight = new GpuTensor(config.dim, 1, QuantMode),

            wq = new GpuTensor(config.dim, config.dim, QuantMode),
            wk = new GpuTensor(config.dim, config.dim, QuantMode),
            wv = new GpuTensor(config.dim, config.dim, QuantMode),
            wo = new GpuTensor(config.dim, config.dim, QuantMode),

            w1 = new GpuTensor(config.hidden_dim, config.dim, QuantMode),
            w2 = new GpuTensor(config.dim, config.hidden_dim, QuantMode),
            w3 = new GpuTensor(config.hidden_dim, config.dim, QuantMode),
          };
        }

        // Read token_embedding_table
        ReadTensor(br, weights.token_embedding_table);
        
        // Read rms_att_weight for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].rms_att_weight);
        }

        // Read wq for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].wq);
        }

        // Read wk for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].wk);
        }

        // Read wv for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].wv);
        }

        // Read wo for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].wo);
        }

        // Read rms_ffn_weight for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].rms_ffn_weight);
        }

        // Read w1 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].w1);
        }

        // Read w2 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].w2);
        }

        // Read w3 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadTensor(br, weights.layerWeights[layer].w3);
        }

        // Read remaining weights
        ReadTensor(br, weights.rms_final_weight);

        // Skip over weights formerly associated with freq_cis_real and freq_cis_imag
        int freqWeightsSize = config.seq_len * config.head_size * sizeof(float);
        br.BaseStream.Position += freqWeightsSize;

        // Read wcls
        if (hasClassifierWeights) {
          ReadTensor(br, weights.wcls);
        }
      }
      
      LoadTokenizer(config.vocab_size);

      Debug.Log("Weights loaded successfully in " + (Time.realtimeSinceStartup - startTime) + "s");
      return (config, weights, Tokenizer);
    }
    catch (Exception e) {
      Debug.LogError($"Failed to load weights from {fullPath}: {e}");
      return (null, null, null); // Failed to load
    }
    finally {
      foreach (IDisposable disposable in _garbage) {
        disposable.Dispose();
      }
      _garbage.Clear();
      Profiler.EndSample();
    }
  }

  private void ReadTensor(BinaryReader br, GpuTensor tensor) {
    int sourceSizeBytes = (int)tensor.Size * sizeof(float);
    NativeArray<byte> sourceArray = new NativeArray<byte>(sourceSizeBytes, Allocator.Persistent);
    byte[] buffer = br.ReadBytes(sourceSizeBytes);

    unsafe {
      fixed (byte* pBuffer = &buffer[0]) {
        UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafePtr(sourceArray), pBuffer, sourceSizeBytes);
      }
    }

    ComputeUtils.Quantize(QuantizationModes.Float32, tensor.Mode, sourceArray, tensor.Buffer);
    
    /*
    if (QuantMode == QuantizationModes.Q8_0) {
      ComputeBuffer checkWeightsBuffer = ComputeUtils.CreateVectorizedBuffer(tensor.Size, QuantizationModes.Float32);
      ComputeUtils.Dequantize(QuantMode, QuantizationModes.Float32,
        tensor.Buffer, checkWeightsBuffer);
      float[] checkWeightsData = new float[checkWeightsBuffer.ElementCount<float>()];
      checkWeightsBuffer.GetData(checkWeightsData);
      NativeArray<float> sourceArrayFloat = sourceArray.Reinterpret<float>(1);
      float maxError = 0;
      float avgError = 0;
      for (int i = 0; i < checkWeightsData.Length; ++i) {
        float error = Mathf.Abs(sourceArrayFloat[i] - checkWeightsData[i]);
        maxError = Mathf.Max(maxError, error);
        avgError += error;
      }

      avgError /= checkWeightsData.Length;
    }
    */

    _garbage.Add(sourceArray);
  }

  public void LoadTokenizer(int vocabSize) {
    string fullPath = VocabPath;
    if (!File.Exists(fullPath)) {
      fullPath = Path.Combine(Application.streamingAssetsPath, "models", fullPath);
    }

    using (BinaryReader reader = new BinaryReader(File.Open(fullPath, FileMode.Open))) {
      int maxTokenLength = (int)reader.ReadUInt32();
      var textToToken = new Dictionary<string, int>(vocabSize);
      string[] tokenToText = new string[vocabSize];
      float[] tokenToScore = new float[vocabSize];
      int sos = 1;
      int eos = 2;

      for (int i = 0; i < vocabSize; i++) {
        tokenToScore[i] = reader.ReadSingle();
        int len = reader.ReadInt32();
        var bytes = reader.ReadBytes(len);
        string vocabEntry = Encoding.UTF8.GetString(bytes);
        textToToken[vocabEntry] = i;
        tokenToText[i] = vocabEntry;

        if (vocabEntry.Trim() == "<s>") {
          sos = i;
        }
        else if (vocabEntry.Trim() == "</s>") {
          eos = i;
        }
      }

      Tokenizer = new Tokenizer(textToToken, tokenToText, tokenToScore, vocabSize, sos, eos);
    }

    Debug.Log($"Loaded {vocabSize} tokens from {VocabPath}");
  }
}