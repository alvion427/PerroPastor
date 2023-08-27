using System;
using System.IO;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

public class GGMLLoader : ModelLoaderBase {
  public static GGMLMetaData LoadMetadata(string path) {
    FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read);
    BinaryReader reader = new BinaryReader(stream);

    GGMLMetaData metaData = new GGMLMetaData();
    metaData.FileVersion = ReadMagic(reader);
    metaData.Hparams = ReadHParams(reader);
    metaData.Vocab = ReadVocab(reader, metaData.Hparams.NVocab);
    metaData.Tensors = ReadTensorMetadata(reader, metaData.FileVersion);
    metaData.NamedTensors = metaData.Tensors.ToDictionary(t => t.Name, t => t);

    return metaData;
  }

  public static LlamaConfig CreateConfig(GGMLMetaData metaData, QuantizationModes weightsMode,
    QuantizationModes runtimeMode) {
    // We don't currently support having different precisions for different weights (though we should!)
    // For now, just assume that the embedding weights are the precision we want for all weights.  In
    // the 16 bit models I've looked at, they are using fp32 for the normalization weights (why?) but we 
    // will just quantize that down on the fly.
    QuantizationModes sourceMode = metaData.NamedTensors["tok_embeddings.weight"].Type.GetTraits().QuantizationMode;

    // Found this magic math to compute the ffn dimension in the llama.cpp code
    // ref: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L194-L199
    uint n_ff_raw = 2 * (4 * metaData.Hparams.NEmbed) / 3;
    uint nmult = metaData.Hparams.NMult;
    uint n_ff = ((n_ff_raw + nmult - 1) / nmult) * nmult;

    // This doesn't seem to come from the model at all, but just be specified by the user.  This must be due to the fact
    // that the context length can change with rope, but we might need to make some changes to support that.
    uint n_ctx = 512;

    return new LlamaConfig(sourceMode, weightsMode, runtimeMode) {
      dim = (int)metaData.Hparams.NEmbed,
      hidden_dim = (int)n_ff,
      n_layers = (int)metaData.Hparams.NLayer,
      n_heads = (int)metaData.Hparams.NHead,
      n_kv_heads = (int)metaData.Hparams.NHeadKv,
      vocab_size = (int)metaData.Hparams.NVocab,
      seq_len = (int)n_ctx,
    };
  }

  protected async override Task<(LlamaConfig, Weights, Tokenizer)> LoadModelImpl(
    QuantizationModes weightQuantMode, QuantizationModes runtimeQuantMode) {
    string path = ModelPath;

    long start = DateTime.Now.Ticks;
    var metaData = await Task.Run(() => { return LoadMetadata(path); });

    bool hasClassifierWeights = metaData.NamedTensors.ContainsKey("output.weight");
    LlamaConfig config = CreateConfig(metaData, weightQuantMode, runtimeQuantMode);
    Weights weights = new Weights(config, weightQuantMode, true);

    using (var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open))
    using (var accessor = mmf.CreateViewAccessor(0, 0)) {
      unsafe {
        byte* fileStart = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref fileStart);
        try {
          LoadIntoNativeArray(weights.token_embedding_table, metaData.NamedTensors["tok_embeddings.weight"], fileStart);
          LoadIntoNativeArray(weights.rms_final_weight, metaData.NamedTensors["norm.weight"], fileStart);
          if (hasClassifierWeights) {
            LoadIntoNativeArray(weights.wcls, metaData.NamedTensors["output.weight"], fileStart);
          }

          for (int l = 0; l < config.n_layers; ++l) {
            void loadLw(NativeArray<byte> dest, string tensorName) {
              tensorName = string.Format(tensorName, l);
              LoadIntoNativeArray(dest, metaData.NamedTensors[tensorName], fileStart);
            }

            var lw = weights.layerWeights[l];
            loadLw(lw.rms_att_weight, "layers.{0}.attention_norm.weight");
            loadLw(lw.rms_ffn_weight, "layers.{0}.ffn_norm.weight");

            loadLw(lw.wq, "layers.{0}.attention.wq.weight");
            loadLw(lw.wk, "layers.{0}.attention.wk.weight");
            loadLw(lw.wv, "layers.{0}.attention.wv.weight");
            loadLw(lw.wo, "layers.{0}.attention.wo.weight");

            loadLw(lw.w1, "layers.{0}.feed_forward.w1.weight");
            loadLw(lw.w2, "layers.{0}.feed_forward.w2.weight");
            loadLw(lw.w3, "layers.{0}.feed_forward.w3.weight");
          }
        }
        finally {
          accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        }
      }
    }

    long end = DateTime.Now.Ticks;
    Debug.Log($"Loaded GGML model in {(end - start) / 10000.0f} ms");

    const int sos = 1;
    const int eos = 2;
    Tokenizer tokenizer = new Tokenizer(metaData.Vocab.TokenToId, metaData.Vocab.IdToToken, metaData.Vocab.IdToScore,
      (int)metaData.Hparams.NVocab, sos, eos);

    return (config, weights, tokenizer);
  }

  private static GGMLFileVersion ReadMagic(BinaryReader reader) {
    uint magic = reader.ReadUInt32();
    if (magic == GGMLFileMagic.GGML) {
      return GGMLFileVersion.GGML;
    }
    else if (magic == GGMLFileMagic.GGMF) {
      uint version = reader.ReadUInt32();
      if (version == 1) {
        return GGMLFileVersion.GGMF_V1;
      }
    }
    else if (magic == GGMLFileMagic.GGJT) {
      uint version = reader.ReadUInt32();
      switch (version) {
        case 1: return GGMLFileVersion.GGJT_V1;
        case 2: return GGMLFileVersion.GGJT_V2;
        case 3: return GGMLFileVersion.GGJT_V3;
      }
    }

    throw new Exception("Unknown magic number");
  }

  private static Hparams ReadHParams(BinaryReader reader) {
    Hparams hparams = new Hparams();
    hparams.NVocab = reader.ReadUInt32();
    hparams.NEmbed = reader.ReadUInt32();
    hparams.NMult = reader.ReadUInt32();
    hparams.NHead = reader.ReadUInt32();
    hparams.NLayer = reader.ReadUInt32();
    hparams.NRot = reader.ReadUInt32();
    hparams.FType = (GGMLFileType)reader.ReadUInt32();

    // LLaMAv2 
    // TODO: Read from header
    hparams.NHeadKv = hparams.NHead;

    return hparams;
  }

  private static Vocab ReadVocab(BinaryReader reader, uint nVocab) {
    Vocab vocab = new Vocab((int)nVocab);

    for (int i = 0; i < nVocab; i++) {
      string token = ReadFixedString(reader);
      float score = reader.ReadSingle();

      vocab.TokenToId[token] = i;
      vocab.IdToToken[i] = token;
      vocab.IdToScore[i] = score;
    }

    return vocab;
  }

  private static string ReadFixedString(BinaryReader reader) {
    int length = (int)reader.ReadUInt32();
    return ReadFixedString(reader, length);
  }

  private static string ReadFixedString(BinaryReader reader, int length) {
    byte[] byteBuffer = reader.ReadBytes(length);
    return System.Text.Encoding.UTF8.GetString(byteBuffer);
  }

  private static List<TensorMeta> ReadTensorMetadata(BinaryReader reader, GGMLFileVersion fileVersion) {
    List<TensorMeta> tensors = new List<TensorMeta>();

    while (reader.BaseStream.Position < reader.BaseStream.Length) {
      TensorMeta tensor = new TensorMeta();

      uint numDims = reader.ReadUInt32();
      uint nameLen = reader.ReadUInt32();
      tensor.Type = (GGMLType)reader.ReadUInt32();

      tensor.Dimensions = new int[numDims];
      for (int i = 0; i < numDims; i++) {
        tensor.Dimensions[i] = (int)reader.ReadUInt32();
      }

      tensor.Name = ReadFixedString(reader, (int)nameLen);

      if (fileVersion >= GGMLFileVersion.GGJT_V1) {
        // Align to 32 bytes
        reader.BaseStream.Position += (32 - (reader.BaseStream.Position % 32));
      }

      tensor.FileOffset = reader.BaseStream.Position;
      tensor.Size = CalculateTensorSize(tensor);
      reader.BaseStream.Position += tensor.Size;

      tensors.Add(tensor);
    }

    return tensors;
  }

  private static long CalculateTensorSize(TensorMeta tensor) {
    var traits = tensor.Type.GetTraits();
    int size = traits.TypeSize;
    foreach (int dim in tensor.Dimensions) {
      size *= dim;
    }

    return size / traits.BlockSize;
  }
  
  private static unsafe void LoadIntoNativeArray(NativeArray<byte> array, TensorMeta tensor, byte* fileStart) {
    if (tensor.Type == GGMLType.F16) {
      byte* tensorPtr = fileStart + tensor.FileOffset;
      if (array.Length != tensor.Size) {
        Debug.LogError($"Mismatched tensor size for {tensor.Name} expected {tensor.Size} but got {array.Length}");
        return;
      }

      UnsafeUtility.MemCpy(array.GetUnsafePtr(), tensorPtr, array.Length);
    }
    else if (tensor.Type == GGMLType.F32) {
      {
        half* arrayPtr = (half*)array.GetUnsafePtr();
        float* tensorPtr = (float*)(fileStart + tensor.FileOffset);
        Debug.Assert(array.Length == tensor.Size / 2);
        int count = (int)(tensor.Size / sizeof(float));
        for (int i = 0; i < count; ++i) {
          arrayPtr[i] = (half)tensorPtr[i];
        }
      }
    }
    else {
      Debug.LogError($"Can't load unknown tensor type: {tensor.Type}");
    }
  }

}