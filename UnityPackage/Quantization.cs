using System;
using System.Collections.Generic;
using UnityEngine;

public enum QuantizationModes {
  // unquantized
  Float32,
  Float16,
  
  // ggml types
  Q8_0,   
}

public static class QuantizationUtil {
  public static int BlockSize(this QuantizationModes quantMode) {
    return BlockSizes[quantMode];
  }

  public static int BlockSizeBytes(this QuantizationModes quantMode) {
    return BlockByteSizes[quantMode];
  }

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

  public static int ElementSize(this QuantizationModes quantMode) {
    if (quantMode == QuantizationModes.Float32) {
      return 4;
    }
    else if (quantMode == QuantizationModes.Float16) {
      return 2;
    }
    else {
      throw new ArgumentException("Quantized buffers don't have an 'ElementSize'");
    }
  }
  
  private static readonly Dictionary<QuantizationModes, string> QuantizationFlags =
    new Dictionary<QuantizationModes, string>() {
      { QuantizationModes.Float32, "QUANT_{0}_32" },
      { QuantizationModes.Float16, "QUANT_{0}_16" },
      { QuantizationModes.Q8_0, "QUANT_{0}_Q8_0"}
    };
  
  private static readonly Dictionary<QuantizationModes, int> BlockSizes = 
    new Dictionary<QuantizationModes, int>() {
      { QuantizationModes.Float32, 4 },
      { QuantizationModes.Float16, 4 },
      { QuantizationModes.Q8_0, 32}
    };

  private static readonly Dictionary<QuantizationModes, int> BlockByteSizes = 
    new Dictionary<QuantizationModes, int>() {
      { QuantizationModes.Float32, 16 },
      { QuantizationModes.Float16, 8 },
      { QuantizationModes.Q8_0, 36}
    };
  
  public static unsafe float[] DequantizeCpu(ComputeBuffer quantizedBuffer, QuantizationModes mode, 
    int offset = 0, int length = 0) {
    int numBlocks = quantizedBuffer.count;
    int blockSize = mode.BlockSize();
    int elementCount = numBlocks * blockSize;

    length = length == 0 ? numBlocks - offset : length; 

    float[] result = new float[elementCount];
    if (mode == QuantizationModes.Q8_0) {
      Q8_0Block[] quantizedData = new Q8_0Block[quantizedBuffer.count];
      quantizedBuffer.GetData(quantizedData);

      if (offset + length > elementCount) {
        throw new ArgumentException($"Not enough space in buffer: {offset} + {length} > {elementCount}");
      }
      for (int b = offset; b < offset + length; ++b) {
        float scale = quantizedData[b].scale;
        for (int i = 0; i < blockSize; ++i) {
          int idx = (b - offset) * blockSize + i;
          result[idx] = quantizedData[b].values[i] * scale;
        }
      }
    }
    else {
      throw new ArgumentException("No dequantize implementation for " + mode);
    }

    return result;
  }
}

