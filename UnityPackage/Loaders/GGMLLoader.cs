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

  public static LlamaConfig CreateConfig(GGMLMetaData metaData) {
    // Found this magic math to compute the ffn dimension in the llama.cpp code
    // ref: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/model.py#L194-L199
    uint n_ff_raw = 2 * (4 * metaData.Hparams.NEmbed) / 3;
    uint nmult = metaData.Hparams.NMult;
    uint n_ff = ((n_ff_raw + nmult - 1) / nmult) * nmult;

    // This doesn't seem to come from the model at all, but just be specified by the user.  This must be due to the fact
    // that the context length can change with rope, but we might need to make some changes to support that.
    uint n_ctx = 512;

    return new LlamaConfig() {
      dim = (int)metaData.Hparams.NEmbed,
      hidden_dim = (int)n_ff,
      n_layers = (int)metaData.Hparams.NLayer,
      n_heads = (int)metaData.Hparams.NHead,
      n_kv_heads = (int)metaData.Hparams.NHeadKv,
      vocab_size = (int)metaData.Hparams.NVocab,
      seq_len = (int)n_ctx,
    };
  }

  protected async override Task<(LlamaConfig, WeightsGpu, Tokenizer)> LoadModelImpl() {
    string path = ModelPath;

    long start = DateTime.Now.Ticks;
    var metaData = await Task.Run(() => { return LoadMetadata(path); });

    bool hasClassifierWeights = metaData.NamedTensors.ContainsKey("output.weight");
    LlamaConfig config = CreateConfig(metaData);
    WeightsGpu weights = new WeightsGpu();
    weights.layerWeights = new LayerWeightsGPU[config.n_layers];

    unsafe {
      var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
      var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
      byte* fileStart = null;
      accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref fileStart);
      try {
        weights.token_embedding_table = CreateAndLoadTensor(metaData.NamedTensors["tok_embeddings.weight"], fileStart);

#if false
        float[] dequantized = QuantizationUtil.DequantizeCpu(weights.token_embedding_table.Buffer,
          weights.token_embedding_table.Mode,
          0, 1);

          throw new Exception();
#endif
        
        weights.rms_final_weight = CreateAndLoadTensor(metaData.NamedTensors["norm.weight"], fileStart);
        if (hasClassifierWeights) {
          weights.wcls = CreateAndLoadTensor(metaData.NamedTensors["output.weight"], fileStart);
        }

        for (int l = 0; l < config.n_layers; ++l) {
          GpuTensor loadLayerTensor(string tensorName) {
            tensorName = string.Format(tensorName, l);
            return CreateAndLoadTensor(metaData.NamedTensors[tensorName], fileStart);
          }

          var lw = new LayerWeightsGPU();
          lw.rms_att_weight = loadLayerTensor("layers.{0}.attention_norm.weight");
          lw.rms_ffn_weight = loadLayerTensor("layers.{0}.ffn_norm.weight");

          lw.wq = loadLayerTensor("layers.{0}.attention.wq.weight");
          lw.wk = loadLayerTensor("layers.{0}.attention.wk.weight");
          lw.wv = loadLayerTensor("layers.{0}.attention.wv.weight");
          lw.wo = loadLayerTensor("layers.{0}.attention.wo.weight");

          lw.w1 = loadLayerTensor("layers.{0}.feed_forward.w1.weight");
          lw.w2 = loadLayerTensor("layers.{0}.feed_forward.w2.weight");
          lw.w3 = loadLayerTensor("layers.{0}.feed_forward.w3.weight");
          weights.layerWeights[l] = lw;
        }
        
        Debug.Log("All layers loaded successfully");
      }
      finally {
        accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        accessor.Dispose();
        mmf.Dispose();
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

  private unsafe GpuTensor CreateAndLoadTensor(TensorMeta tensorMeta, byte* fileStart) {
    QuantizationModes quantMode;
    switch (tensorMeta.Type) {
      case GGMLType.F32:
        quantMode = QuantizationModes.Float32;
        break;
      case GGMLType.F16:
        quantMode = QuantizationModes.Float16;
        break;
      case GGMLType.Q5_1:
        quantMode = QuantizationModes.Q5_1;
        break;
      case GGMLType.Q8_0:
        quantMode = QuantizationModes.Q8_0;
        break;
      default:
        throw new ArgumentException("Unsupported tensorMeta type: " + tensorMeta.Type);
    }

    /*
     TODO: Verify this
    if (tensorMeta.Dimensions.Length == 1) {
      if (rows != tensorMeta.Dimensions[0] || cols > 1) {
        throw new ArgumentException($"tensorMeta size doesn't match: {rows}, {cols} != {tensorMeta.Dimensions}");
      }
    }
    if (tensorMeta.Dimensions.Length == 2) {
      if (rows != tensorMeta.Dimensions[0] || cols != tensorMeta.Dimensions[1]) {
        throw new ArgumentException($"tensorMeta size doesn't match: {rows}, {cols} != {tensorMeta.Dimensions}");
      }
    }
    else  {
      throw new ArgumentException("tensorMetas of more than 2 dimensions not supported");
    }
    */
    
    int rows = tensorMeta.Dimensions[0];
    int cols = tensorMeta.Dimensions.Length >= 2 ? tensorMeta.Dimensions[1] : 1; 
    GpuTensor tensor = new GpuTensor(rows, cols, quantMode);

    byte* tensorPtr = fileStart + tensorMeta.FileOffset;

    if (tensor.Mode == QuantizationModes.Q8_0) {
      // *sigh*, Q8_0 buffers have a stride of size 34 from the fact that they use a 'half' for scale.  ComputeBuffers
      // don't allow strides that aren't divisible by 4, so we use a full float for scale.  That means though that we 
      // have to convert the data at load time.
      NativeArray<Q8_0Block> blockArray = new NativeArray<Q8_0Block>(tensor.BlockCount, Allocator.Temp);
      ConvertQ8_0Buffer(tensorPtr, blockArray);
      tensor.Buffer.SetData(blockArray);
    }
    else {
      if (tensor.SizeBytes != tensorMeta.Size) { 
        throw new ArgumentException($"Mismatched tensor size for {tensorMeta.Name} expected {tensorMeta.Size} but got {tensor.SizeBytes}");
      }
    
      var sourceArray =
        NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<byte>(tensorPtr, (int)tensor.SizeBytes, Allocator.None);

      var safety = AtomicSafetyHandle.Create();
      NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref sourceArray, safety);
      
      tensor.Buffer.SetData(sourceArray);
    }

    return tensor;
  }

  private unsafe void ConvertQ8_0Buffer(byte* tensorPtr, NativeArray<Q8_0Block> destArray) {
    Q8_0Block* destPtr = (Q8_0Block*)NativeArrayUnsafeUtility.GetUnsafePtr(destArray);
    int blockSize = QuantizationModes.Q8_0.BlockSize();
    int sourceSize = GGMLType.Q8_0.GetTraits().TypeSize;
    //for (int b = 0; b < destArray.Length; ++b) {
    Parallel.For(0, destArray.Length, b => {
      byte* sourceBlockPtr = tensorPtr + (b * sourceSize);
      destPtr[b].scale = *(half*)sourceBlockPtr;
      UnsafeUtility.MemCpy(destPtr[b].values, sourceBlockPtr + 2, blockSize);
    });
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
    long size = traits.TypeSize;
    foreach (int dim in tensor.Dimensions) {
      size *= dim;
    }

    return size / traits.BlockSize;
  }
}