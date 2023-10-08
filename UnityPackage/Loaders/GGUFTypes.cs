using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

public class GGUFMetaData {
  public int Version;
  public int Alignment;
  public long DataStart;
  
  public Dictionary<string, object> KeyValues;
  public List<TensorMeta> Tensors;
  public Dictionary<string, TensorMeta> NamedTensors;
  
  public T GetValue<T>(string key) {
    return (T)KeyValues[key];
  }
}

public struct TensorMeta 
{
  public string Name {get; set;}
  public int[] Dimensions {get; set;}
  public GGUFType Type {get; set;}
  public long Size {get; set;}
  public long FileOffset {get; set;}
}

public enum GGUFType {
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

public unsafe struct Q8_0Block {
  public float scale;
  public fixed sbyte values[32];
}

public unsafe struct Q5_1Block
{
  public half scale;
  public half min;
  public uint highBits;
  public fixed uint values[4];
};


public class GGUFTraitType 
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
  public GGUFType VecDotType { get; set; }

  public static readonly GGUFTraitType F32 = new GGUFTraitType
  {
    TypeName = "f32",
    BlockSize = 1,
    TypeSize = 4,
    IsQuantized = false,
    QuantizationMode = QuantizationModes.Float32,
  };

  public static readonly GGUFTraitType F16 = new GGUFTraitType
  {
    TypeName = "f16",
    BlockSize = 1,
    TypeSize = 2,
    IsQuantized = false,
    QuantizationMode = QuantizationModes.Float16,
  };
  
  public static readonly GGUFTraitType Q5_1 = new GGUFTraitType
  {
    TypeName = "q5_1",
    BlockSize = 32,
    TypeSize = 24,  // 32 4 bit nibbles plus 16 high bits plus two bytes each for min/max
    IsQuantized = true,
    QuantizationMode = QuantizationModes.Q5_1,
  };

  public static readonly GGUFTraitType Q8_0 = new GGUFTraitType
  {
    TypeName = "q8_0",
    BlockSize = 32,
    TypeSize = 34,  // 32 byte values + 2 for half scale (note this is different that our runtime type)
    IsQuantized = true,
    QuantizationMode = QuantizationModes.Q8_0,
  };

  public static readonly Dictionary<GGUFType, GGUFTraitType> Traits = new Dictionary<GGUFType, GGUFTraitType>()
  {
    {GGUFType.F32, F32},
    {GGUFType.F16, F16},
    { GGUFType.Q5_1, Q5_1},
    { GGUFType.Q8_0, Q8_0},
  };
}

public static class GGUFTraitTypeExtensions {
  public static GGUFTraitType GetTraits(this GGUFType type) {
    return GGUFTraitType.Traits[type];
  }
}
