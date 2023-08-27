using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GGMLMetaData {
  public GGMLFileVersion FileVersion;
  public Hparams Hparams;
  public Vocab Vocab;
  public List<TensorMeta> Tensors;
  public Dictionary<string, TensorMeta> NamedTensors;
}

public struct Hparams
{
  public uint NVocab;
  public uint NEmbed;
  public uint NMult;
  public uint NHead;
  public uint NLayer;
  public uint NRot;
  public GGMLFileType FType { get; set; }
  public uint NHeadKv { get; set; }
}

public class Vocab {
  public Vocab(int vocabSize) {
    TokenToId = new Dictionary<string, int>(vocabSize);
    IdToToken = new string[vocabSize];
    IdToScore = new float[vocabSize];
  }

  public Dictionary<string, int> TokenToId;
  public string[] IdToToken;
  public float[] IdToScore;
}

public struct TensorMeta 
{
  public string Name {get; set;}
  public int[] Dimensions {get; set;}
  public GGMLType Type {get; set;}
  public long Size {get; set;}
  public long FileOffset {get; set;}
}

public enum GGMLType 
{
  F32,
  F16,
  Q4_0,
  Q4_1,
  Q5_0, 
  Q5_1,
  Q8_0,
  Q8_1,
  
  // k-quantizations
  Q2_K,
  Q3_K,  
  Q4_K,
  Q5_K,
  Q6_K,
  Q8_K,

  I8,
  I16,
  I32,
  
  Count
}

public static class GGMLFileMagic {

  public const uint GGJT = 0x67676a74; // 'ggjt'
  public const uint GGLA = 0x67676c61; // 'ggla'  
  public const uint GGMF = 0x67676d66; // 'ggmf'
  public const uint GGML = 0x67676d6c; // 'ggml'
  public const uint GGSN = 0x6767736e; // 'ggsn'
}

public enum GGMLFileVersion {
  GGML,
  GGMF_V1,
  GGJT_V1, 
  GGJT_V2,
  GGJT_V3
}

public enum GGMLFileType
{
  AllFloat32, // all f32
  MostlyFloat16, // except 1d tensors
  MostlyQ4_0, // except 1d tensors
  MostlyQ4_1, // except 1d tensors
  MostlyQ4_1SomeFloat16, // tok_embeddings.weight and output.weight are F16
  MostlyQ8_0, // except 1d tensors
  MostlyQ5_0, // except 1d tensors
  MostlyQ5_1, // except 1d tensors
  MostlyQ2_K, // except 1d tensors
  MostlyQ3_K_Small, // except 1d tensors
  MostlyQ3_K_Medium, // except 1d tensors 
  MostlyQ3_K_Large, // except 1d tensors
  MostlyQ4_K_Small, // except 1d tensors
  MostlyQ4_K_Medium, // except 1d tensors
  MostlyQ5_K_Small, // except 1d tensors
  MostlyQ5_K_Medium, // except 1d tensors
  MostlyQ6_K // except 1d tensors
}

public class GGMLTraitType 
{
  public string TypeName ;
  public int BlockSize ;
  public int TypeSize ;
  public bool IsQuantized ;

  public QuantizationModes QuantizationMode;

  public Func<float[], float[]> ToFloat { get; set; }
  public Func<float[], byte[]> FromFloat { get; set; }
  public Func<float[], byte[]> FromFloatReference { get; set; }

  public Func<byte[], byte[], float> VecDot { get; set; }
  public GGMLType VecDotType { get; set; }

  public static readonly GGMLTraitType F32 = new GGMLTraitType
  {
    TypeName = "f32",
    BlockSize = 1,
    TypeSize = 4,
    IsQuantized = false,
    QuantizationMode = QuantizationModes.Float32,
    // Populate methods
  };

  public static readonly GGMLTraitType F16 = new GGMLTraitType
  {
    TypeName = "f16",
    BlockSize = 1,
    TypeSize = 2,
    IsQuantized = false,
    QuantizationMode = QuantizationModes.Float16,
    // Populate methods
  };

  public static readonly Dictionary<GGMLType, GGMLTraitType> Traits = new Dictionary<GGMLType, GGMLTraitType>()
  {
    {GGMLType.F32, F32},
    {GGMLType.F16, F16},
  };
}

public static class GGMLTraitTypeExtensions {
  public static GGMLTraitType GetTraits(this GGMLType type) {
    return GGMLTraitType.Traits[type];
  }
}
