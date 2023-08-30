using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

public class QuantizeTests : MonoBehaviour {
  public bool RunTest = true;
  public float Scale = 1;
  public int Count = 256;

  private void Update() {
    if (RunTest) {
      RunTest = false;
      
      float[] input = Enumerable.Range(0, Count).Select(i => (float)Random.Range(0, Scale * 2.0f) - Scale).ToArray();
      NativeArray<float> inputArray = new NativeArray<float>(Count, Allocator.Temp);
      inputArray.CopyFrom(input);

      ComputeBuffer quantizedBuffer = ComputeUtils.CreateBlockBuffer(Count, QuantizationModes.Q8_0);

      ComputeUtils.Quantize(QuantizationModes.Float32, QuantizationModes.Q8_0, 
        inputArray.Reinterpret<byte>(sizeof(float)), quantizedBuffer);

      float[] checkData = QuantizationUtil.DequantizeCpu(quantizedBuffer, QuantizationModes.Q8_0);
      {
        float avgError = 0;
        float maxError = 0;
        for (int i = 0; i < Count; ++i) {
          float error = Mathf.Abs(inputArray[i] - checkData[i]);
          avgError += error;
          maxError = Mathf.Max(maxError, error);
        }

        avgError /= Count;

        Debug.Log($"Computed quantization with scale avg error {avgError}, and max error {maxError}");
        Debug.Log(string.Join(", ", checkData));
      }

      ComputeBuffer dequantizedBuffer = new ComputeBuffer(Count, sizeof(float));
      ComputeUtils.Dequantize(QuantizationModes.Q8_0, QuantizationModes.Float32, 
        quantizedBuffer, dequantizedBuffer);

      QuantizationUtil.DequantizeCpu(quantizedBuffer, QuantizationModes.Q8_0);

      float[] dequantizedData = new float[Count];
      dequantizedBuffer.GetData(dequantizedData);

      {
        float avgError = 0;
        float maxError = 0;
        for (int i = 0; i < Count; ++i) {
          float error = Mathf.Abs(inputArray[i] - dequantizedData[i]);
          avgError += error;
          maxError = Mathf.Max(maxError, error);
        }

        avgError /= Count;

        Debug.Log($"Computed DE-quantization avg error {avgError}, and max error {maxError}");
        Debug.Log(string.Join(", ", dequantizedData));
      }
    }
  }
}
