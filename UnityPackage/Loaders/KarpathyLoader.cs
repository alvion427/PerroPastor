using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.VisualScripting.YamlDotNet.Core.Tokens;
using UnityEngine;
using UnityEngine.Profiling;

public class KarpathyLoader : ModelLoaderBase {
  public string VocabPath;
  public LlamaConfig Config;
  public Weights Weights;


  protected async override Task<(LlamaConfig, Weights, Tokenizer)> LoadModelImpl(
    QuantizationModes weightQuantMode, QuantizationModes runtimeQuantMode) {
    if (!File.Exists(ModelPath)) {
      ModelPath = Path.Combine(Application.streamingAssetsPath, "models", ModelPath);
    }

    // Karpathy's format only uses float32
    const QuantizationModes sourceMode = QuantizationModes.Float32;

    float startTime = Time.realtimeSinceStartup;
    Debug.Log("Loading weights...");
    try {
      Profiler.BeginSample("LoadWeights");
      LlamaConfig config = null;
      Weights weights = null;

      using (FileStream fs = new FileStream(ModelPath, FileMode.Open, FileAccess.Read))
      using (BinaryReader br = new BinaryReader(fs)) {
        // Read the config
        config = new LlamaConfig(sourceMode, weightQuantMode, runtimeQuantMode) {
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
        weights = new Weights(config, QuantizationModes.Float32, hasClassifierWeights);

        // Read token_embedding_table
        ReadNativeArray(br, weights.token_embedding_table);

        // Read rms_att_weight for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].rms_att_weight);
        }

        // Read wq for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].wq);
        }

        // Read wk for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].wk);
        }

        // Read wv for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].wv);
        }

        // Read wo for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].wo);
        }

        // Read rms_ffn_weight for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].rms_ffn_weight);
        }

        // Read w1 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].w1);
        }

        // Read w2 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].w2);
        }

        // Read w3 for all layers
        for (int layer = 0; layer < config.n_layers; layer++) {
          ReadNativeArray(br, weights.layerWeights[layer].w3);
        }

        // Read remaining weights
        ReadNativeArray(br, weights.rms_final_weight);

        // Skip over weights formerly associated with freq_cis_real and freq_cis_imag
        int freqWeightsSize = config.seq_len * config.head_size * sizeof(float);
        br.BaseStream.Position += freqWeightsSize;

        // Read wcls
        if (hasClassifierWeights) {
          ReadNativeArray(br, weights.wcls);
        }
      }
      
      LoadTokenizer(config.vocab_size);

      Debug.Log("Weights loaded successfully in " + (Time.realtimeSinceStartup - startTime) + "s");
      return (config, weights, Tokenizer);
    }
    catch (Exception e) {
      Debug.LogError($"Failed to load weights from {ModelPath}: {e}");
      return (null, null, null); // Failed to load
    }
    finally {
      Profiler.EndSample();
    }
  }

  private void ReadNativeArray(BinaryReader br, NativeArray<byte> byteArray) {
    NativeArray<float> array = byteArray.Reinterpret<float>(sizeof(byte));
    int byteCount = array.Length * sizeof(float);
    byte[] buffer = br.ReadBytes(byteCount);

    // Get a pointer to the NativeArray's data
    unsafe {
      fixed (byte* pBuffer = &buffer[0]) {
        UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafePtr(array), pBuffer, byteCount);
      }
    }
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