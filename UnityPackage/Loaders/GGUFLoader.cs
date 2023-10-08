using System;
using System.IO;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine;

public class GGUFLoader : ModelLoaderBase {
  
  public bool CacheWeights = false;

  private class CachedWeights {
    public string ModelPath;
    public LlamaConfig Config;
    public WeightsGpu Weights;
    public Tokenizer Tokenizer;
  }
  
  private static CachedWeights _cachedWeights;

  static GGUFLoader() {
#if UNITY_EDITOR
    AssemblyReloadEvents.beforeAssemblyReload += () => {
      if (_cachedWeights != null) {
        _cachedWeights.Weights.RemoveReference();
        _cachedWeights = null;
      }
    };
#endif
  }
  
  
  enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
  };


  public static GGUFMetaData LoadMetadata(string path) {
    FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read);
    BinaryReader reader = new BinaryReader(stream);

    const uint kGGUF_MAGIC = 0x46554747;

    uint magic = reader.ReadUInt32();
    if (magic != kGGUF_MAGIC) {
      throw new Exception("Unknown magic number: " + magic);
    }

    GGUFMetaData metaData = new GGUFMetaData();
    metaData.Version = reader.ReadInt32();
    
    if (metaData.Version == 1) {
      throw new Exception("GGUFv1 not supported");
    }

    uint numTensors = (uint)reader.ReadUInt64();
    uint numKv = (uint)reader.ReadUInt64();
    
    metaData.KeyValues = ReadKeyValues(reader, (int)numKv);
    metaData.Tensors = ReadTensorMetadata(reader, numTensors);

    metaData.NamedTensors = metaData.Tensors.ToDictionary(t => t.Name, t => t);

    if (metaData.KeyValues.ContainsKey("general.alignment")) {
      metaData.Alignment = metaData.GetValue<int>("general.alignment");
    }
    else {
      metaData.Alignment = 32;      
    }

    long offsetPadding = metaData.Alignment - stream.Position % metaData.Alignment;
    stream.Position += offsetPadding;
    metaData.DataStart = stream.Position;
    
    return metaData;
  }

  public static Dictionary<string, object> ReadKeyValues(BinaryReader reader, int numKv) {
    Dictionary<string, object> result = new (numKv);

    for (int i = 0; i < numKv; ++i) {
      string key = ReadFixedString(reader);
      gguf_type type = (gguf_type)reader.ReadUInt32();
      object value = null;
      
      T[] ReadArray<T>(Func<T> readFunc) {
        long numElements = (long)reader.ReadUInt64();
        T[] array = new T[numElements];
        for (int j = 0; j < numElements; ++j) {
          array[j] = readFunc();
        }

        return array;
      }

      switch (type) {
        case gguf_type.GGUF_TYPE_UINT8: value = reader.ReadByte(); break;
        case gguf_type.GGUF_TYPE_INT8: value = (sbyte)reader.ReadByte(); break;
        case gguf_type.GGUF_TYPE_UINT16: value = reader.ReadUInt16(); break;
        case gguf_type.GGUF_TYPE_INT16: value = reader.ReadInt16(); break;
        case gguf_type.GGUF_TYPE_UINT32: value = reader.ReadUInt32(); break;
        case gguf_type.GGUF_TYPE_INT32: value = reader.ReadInt32(); break;
        case gguf_type.GGUF_TYPE_FLOAT32: value = reader.ReadSingle(); break;
        case gguf_type.GGUF_TYPE_UINT64: value = reader.ReadUInt64(); break;
        case gguf_type.GGUF_TYPE_INT64: value = reader.ReadInt64(); break;
        case gguf_type.GGUF_TYPE_FLOAT64: value = reader.ReadDouble(); break;
        case gguf_type.GGUF_TYPE_BOOL: value = reader.ReadBoolean(); break;
        case gguf_type.GGUF_TYPE_STRING: value = ReadFixedString(reader); break;
        case gguf_type.GGUF_TYPE_ARRAY:
          gguf_type elementType = (gguf_type)reader.ReadUInt32();
          switch (elementType) {
            case gguf_type.GGUF_TYPE_UINT8: value = ReadArray(reader.ReadByte); break;
            case gguf_type.GGUF_TYPE_INT8: value = ReadArray(() => (sbyte)reader.ReadByte()); break;
            case gguf_type.GGUF_TYPE_UINT16: value = ReadArray(reader.ReadUInt16); break;
            case gguf_type.GGUF_TYPE_INT16: value = ReadArray(reader.ReadInt16); break;
            case gguf_type.GGUF_TYPE_UINT32: value = ReadArray(reader.ReadUInt32); break;
            case gguf_type.GGUF_TYPE_INT32: value = ReadArray(reader.ReadInt32); break;
            case gguf_type.GGUF_TYPE_FLOAT32: value = ReadArray(reader.ReadSingle); break;
            case gguf_type.GGUF_TYPE_UINT64: value = ReadArray(reader.ReadUInt64); break;
            case gguf_type.GGUF_TYPE_INT64: value = ReadArray(reader.ReadInt64); break;
            case gguf_type.GGUF_TYPE_FLOAT64: value = ReadArray(reader.ReadDouble); break;
            case gguf_type.GGUF_TYPE_BOOL: value = ReadArray(reader.ReadBoolean); break;
            case gguf_type.GGUF_TYPE_STRING: value = ReadArray(() => ReadFixedString(reader)); break;
            case gguf_type.GGUF_TYPE_ARRAY:
            default:
              throw new ArgumentException("Invalid array element type: " + elementType);
          }
          break;
          
        default: 
          throw new ArgumentException("Invalid type: " + type);
      }

      if (value != null) {
        result[key] = value;
      }
    }

    return result;
  }
  
  private static List<TensorMeta> ReadTensorMetadata(BinaryReader reader, uint numTensors) {
    List<TensorMeta> tensors = new List<TensorMeta>((int)numTensors);

    long lastEnd = 0;

    for (int i = 0; i < numTensors; ++i) {
      TensorMeta tensor = new TensorMeta();
      tensor.Name = ReadFixedString(reader);

      uint numDims = reader.ReadUInt32();
      tensor.Dimensions = new int[numDims];
      for (int d = 0; d < numDims; d++) {
        tensor.Dimensions[d] = (int)reader.ReadUInt64();
      }

      tensor.Type = (GGUFType)reader.ReadUInt32();
      tensor.FileOffset = (long)reader.ReadUInt64();

      if (tensor.FileOffset != lastEnd) {
        Debug.LogWarning("Tensors don't line up");
      }

      if (!GGUFTraitType.Traits.ContainsKey(tensor.Type)) {
        Debug.LogError($"Tensor {tensor.Name} has unsupported type: {tensor.Type}");
      }
      
      tensor.Size = CalculateTensorSize(tensor);

      lastEnd = tensor.FileOffset + tensor.Size;

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

  public static LlamaConfig CreateConfig(GGUFMetaData metaData) {
    object v;
    return new LlamaConfig() {
      dim = Convert.ToInt32(metaData.KeyValues["llama.embedding_length"]),
      hidden_dim = Convert.ToInt32(metaData.KeyValues["llama.feed_forward_length"]),
      n_layers = Convert.ToInt32(metaData.KeyValues["llama.block_count"]),
      n_heads = Convert.ToInt32(metaData.KeyValues["llama.attention.head_count"]),
      n_kv_heads = Convert.ToInt32(metaData.KeyValues["llama.attention.head_count_kv"]),
      vocab_size = ((string[])metaData.KeyValues["tokenizer.ggml.tokens"]).Length,
      seq_len = Convert.ToInt32(metaData.KeyValues["llama.context_length"]),
      //seq_len = 512,
    };
  }

  protected async override Task<(LlamaConfig, WeightsGpu, Tokenizer)> LoadModelImpl() {
    string fullPath = GetFullModelPath();

    if (CacheWeights && _cachedWeights != null) {
      if (_cachedWeights.ModelPath == fullPath && _cachedWeights.Weights.IsValid()) {
        Debug.Log("Using cached weights at path " + _cachedWeights.ModelPath);
        _cachedWeights.Weights.AddReference();
        return (_cachedWeights.Config, _cachedWeights.Weights, _cachedWeights.Tokenizer);
      }
      else {
        Debug.Log("Freeing old cached weights at path " + _cachedWeights.ModelPath);
        _cachedWeights.Weights.RemoveReference();
        _cachedWeights = null;
      }
    }

    long start = DateTime.Now.Ticks;
    var metaData = await Task.Run(() => { return LoadMetadata(fullPath); });

    bool hasClassifierWeights = metaData.NamedTensors.ContainsKey("output.weight");
    LlamaConfig config = CreateConfig(metaData);
    WeightsGpu weights = new WeightsGpu();
    weights.layerWeights = new LayerWeightsGPU[config.n_layers];

    unsafe {
      var mmf = MemoryMappedFile.CreateFromFile(fullPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
      var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
      byte* fileStart = null;
      accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref fileStart);
      try {
        weights.token_embedding_table = CreateAndLoadTensor(metaData, "token_embd.weight", fileStart);
        
        /*
        ComputeBuffer checkBuffer = new ComputeBuffer((int)weights.token_embedding_table.Size, sizeof(float));
        ComputeUtils.Dequantize(weights.token_embedding_table.Mode, QuantizationModes.Float32, weights.token_embedding_table.Buffer, checkBuffer);
        float[] checkBufferData = new float[checkBuffer.count];
        checkBuffer.GetData(checkBufferData);
        string checkString = string.Join(", ", new ArraySegment<float>(checkBufferData, 0, 32).ToList());
        Debug.Log("Embedding weights: " + checkString);
        checkBuffer.Dispose();
        */

        weights.rms_final_weight = CreateAndLoadTensor(metaData, "output_norm.weight", fileStart);
        if (hasClassifierWeights) {
          weights.wcls = CreateAndLoadTensor(metaData, "output.weight", fileStart);
        }

        for (int l = 0; l < config.n_layers; ++l) {
          GpuTensor loadLayerTensor(string tensorName) {
            tensorName = string.Format(tensorName, l);
            return CreateAndLoadTensor(metaData, tensorName, fileStart);
          }

          var lw = new LayerWeightsGPU();
          lw.rms_att_weight = loadLayerTensor("blk.{0}.attn_norm.weight");
          lw.rms_ffn_weight = loadLayerTensor("blk.{0}.ffn_norm.weight");

          lw.wq = loadLayerTensor("blk.{0}.attn_q.weight");
          lw.wk = loadLayerTensor("blk.{0}.attn_k.weight");
          lw.wv = loadLayerTensor("blk.{0}.attn_v.weight");
          lw.wo = loadLayerTensor("blk.{0}.attn_output.weight");

          lw.w1 = loadLayerTensor("blk.{0}.ffn_gate.weight");
          lw.w2 = loadLayerTensor("blk.{0}.ffn_down.weight");
          lw.w3 = loadLayerTensor("blk.{0}.ffn_up.weight");
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
    Debug.Log($"Loaded GGUF model in {(end - start) / 10000.0f} ms");

    const int sos = 1;
    const int eos = 2;

    Dictionary<string, string> replacementStrings = new Dictionary<string, string>
    {
      { "<s>", "" },
      { "▁", " " },
      { "<0x0A>", "\n" },
    };

    string applyReplacements(string input)
    {
      string output = input;
      foreach (var kvp in replacementStrings)
      {
        output = output.Replace(kvp.Key, kvp.Value);
      }

      // Handle UTF-8 encoded sequences like <0xC3><0x88>
      output = Regex.Replace(output, @"<0x(..)><0x(..)>", m =>
      {
        byte[] bytes = new byte[] { Convert.ToByte(m.Groups[1].Value, 16), Convert.ToByte(m.Groups[2].Value, 16) };
        return Encoding.UTF8.GetString(bytes);
      });

      return output;
    }
    
    string[] tokenToText = ((string[])metaData.KeyValues["tokenizer.ggml.tokens"]).Select(
      s => applyReplacements(s)).ToArray();
    
    for (int i = 0; i < tokenToText.Length; ++i)
    {
      var s = tokenToText[i];
      if (s.Contains('▁')) {
        Debug.LogError($"WEIRD! Token {i} contains ▁: {s}");
      }
    }
    
    float[] tokenToScore = (float[])metaData.KeyValues["tokenizer.ggml.scores"];
    Dictionary<string, int> textToToken = tokenToText.Select((t, i) => new {t, i}).ToDictionary(x => x.t, x => x.i);

    Tokenizer tokenizer = new Tokenizer(textToToken, tokenToText, tokenToScore, tokenToText.Length, sos, eos);

    if (CacheWeights) {
      Debug.Log("Caching weights at path " + fullPath);
      weights.AddReference();
      _cachedWeights = new CachedWeights() {
        ModelPath = fullPath,
        Config = config,
        Weights = weights,
        Tokenizer = tokenizer,
      };
    }
    
    return (config, weights, tokenizer);
  }

  private unsafe GpuTensor CreateAndLoadTensor(GGUFMetaData metaData, string tensorName, byte* fileStart) {
    var tensorMeta = metaData.NamedTensors[tensorName];
    
    QuantizationModes quantMode;
    switch (tensorMeta.Type) {
      case GGUFType.F32:
        quantMode = QuantizationModes.Float32;
        break;
      case GGUFType.F16:
        quantMode = QuantizationModes.Float16;
        break;
      case GGUFType.Q5_1:
        quantMode = QuantizationModes.Q5_1;
        break;
      case GGUFType.Q8_0:
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

    byte* tensorPtr = fileStart + metaData.DataStart + tensorMeta.FileOffset;

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

#if ENABLE_UNITY_COLLECTIONS_CHECKS
      var safety = AtomicSafetyHandle.Create();
      NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref sourceArray, safety);
#endif
      
      tensor.Buffer.SetData(sourceArray);
    }

    return tensor;
  }

  private unsafe void ConvertQ8_0Buffer(byte* tensorPtr, NativeArray<Q8_0Block> destArray) {
    Q8_0Block* destPtr = (Q8_0Block*)NativeArrayUnsafeUtility.GetUnsafePtr(destArray);
    int blockSize = QuantizationModes.Q8_0.BlockSize();
    int sourceSize = GGUFType.Q8_0.GetTraits().TypeSize;
    //for (int b = 0; b < destArray.Length; ++b) {
    Parallel.For(0, destArray.Length, b => {
      byte* sourceBlockPtr = tensorPtr + (b * sourceSize);
      destPtr[b].scale = *(half*)sourceBlockPtr;
      UnsafeUtility.MemCpy(destPtr[b].values, sourceBlockPtr + 2, blockSize);
    });
  }

  private static string ReadFixedString(BinaryReader reader) {
    long len = (long)reader.ReadUInt64();
    byte[] bytes = reader.ReadBytes((int)len);
    return System.Text.Encoding.UTF8.GetString(bytes);
  }
}