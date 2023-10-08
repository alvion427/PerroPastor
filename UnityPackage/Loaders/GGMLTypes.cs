using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

public class GGMLMetaData {
  public GGMLFileVersion FileVersion;
  public Hparams Hparams;
  public Vocab Vocab;
  public List<GGMLTensorMeta> Tensors;
  public Dictionary<string, GGMLTensorMeta> NamedTensors;
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

public struct GGMLTensorMeta 
{
  public string Name {get; set;}
  public int[] Dimensions {get; set;}
  public GGMLType Type {get; set;}
  public long Size {get; set;}
  public long FileOffset {get; set;}
}

public enum GGMLType {
  F32 = 0,
  F16 = 1, 
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  
  // k-quantizations
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  
  I8 = 16,
  I16 = 17,
  I32 = 18,
  
  Count = 19
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
  };

  public static readonly GGMLTraitType F16 = new GGMLTraitType
  {
    TypeName = "f16",
    BlockSize = 1,
    TypeSize = 2,
    IsQuantized = false,
    QuantizationMode = QuantizationModes.Float16,
  };
  
  public static readonly GGMLTraitType Q5_1 = new GGMLTraitType
  {
    TypeName = "q5_1",
    BlockSize = 32,
    TypeSize = 24,  // 32 4 bit nibbles plus 16 high bits plus two bytes each for min/max
    IsQuantized = true,
    QuantizationMode = QuantizationModes.Q5_1,
  };

  public static readonly GGMLTraitType Q8_0 = new GGMLTraitType
  {
    TypeName = "q8_0",
    BlockSize = 32,
    TypeSize = 34,  // 32 byte values + 2 for half scale (note this is different that our runtime type)
    IsQuantized = true,
    QuantizationMode = QuantizationModes.Q8_0,
  };

  public static readonly Dictionary<GGMLType, GGMLTraitType> Traits = new Dictionary<GGMLType, GGMLTraitType>()
  {
    {GGMLType.F32, F32},
    {GGMLType.F16, F16},
    { GGMLType.Q5_1, Q5_1},
    { GGMLType.Q8_0, Q8_0},
  };
}

public static class GGMLTraitTypeExtensions {
  public static GGMLTraitType GetTraits(this GGMLType type) {
    return GGMLTraitType.Traits[type];
  }
}
