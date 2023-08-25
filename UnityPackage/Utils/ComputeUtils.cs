using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

public static class ComputeUtils {
  private static ComputeShader _computeShader;
  private static int _setDataQuantizedKernel = -1;
  private static int _setDataQuantizedInterleavedKernel = -1;

  public static int GetVectorizedLength(int rawLength) {
    Debug.Assert(rawLength % 4 == 0);
    return rawLength / 4;
  }

  public static int ElementCount<T>(this ComputeBuffer buffer) {
    return (buffer.count * buffer.stride) / Marshal.SizeOf<T>();
  }
  
  public static ComputeBuffer CreateVectorizedBuffer(int size, int elementSize) {
      Debug.Assert(size % 4 == 0);
      return new ComputeBuffer(size / 4, elementSize * 4);
  }

  public static void SetQuantizedData(QuantizationModes mode, ComputeBuffer outputBuffer, NativeArray<float> data) {
    if (mode == QuantizationModes.Float32) {
      outputBuffer.SetData(data);
      return;
    }

    int vecLen = GetVectorizedLength(data.Length);

    _loadShader();

    ComputeBuffer stagingBuffer = new ComputeBuffer(data.Length, sizeof(float));
    stagingBuffer.SetData(data);
    _computeShader.SetBuffer(_setDataQuantizedKernel, "setquant_input", stagingBuffer);
    _computeShader.SetBuffer(_setDataQuantizedKernel, "setquant_output", outputBuffer);
    _computeShader.SetInt("setquant_veclen", vecLen);
    QuantizationUtil.EnableQuantizationKeywords(_computeShader, mode, QuantizationModes.Float32);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _computeShader.Dispatch(_setDataQuantizedKernel, threadGroupsX, 1, 1);
    stagingBuffer.Dispose();
  }

  public static void SetQuantizedDataInterleaved(QuantizationModes mode, ComputeBuffer outputBuffer, NativeArray<float> dataA, NativeArray<float> dataB) {
    _loadShader();

    int count = dataA.Length;
    int vecLen = count / 2;

    // Load into staging buffers
    ComputeBuffer stagingBufferA = new ComputeBuffer(count, sizeof(float));
    ComputeBuffer stagingBufferB = new ComputeBuffer(count, sizeof(float));
    stagingBufferA.SetData(dataA);
    stagingBufferB.SetData(dataB);
    
    _computeShader.SetBuffer(_setDataQuantizedInterleavedKernel, "setquant_inputA", stagingBufferA);
    _computeShader.SetBuffer(_setDataQuantizedInterleavedKernel, "setquant_inputB", stagingBufferB);
    _computeShader.SetBuffer(_setDataQuantizedInterleavedKernel, "setquant_output", outputBuffer);
    _computeShader.SetInt("setquant_veclen", vecLen);
    QuantizationUtil.EnableQuantizationKeywords(_computeShader, mode, QuantizationModes.Float32);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _computeShader.Dispatch(_setDataQuantizedInterleavedKernel, threadGroupsX, 1, 1);
    stagingBufferA.Dispose();
    stagingBufferB.Dispose();
  }

  private static void _loadShader() {
    if (_computeShader == null) {
      _computeShader = Resources.Load<ComputeShader>("ComputeUtils");
      _setDataQuantizedKernel = _computeShader.FindKernel("SetQuantizedData");
      _setDataQuantizedInterleavedKernel = _computeShader.FindKernel("SetQuantizedDataInterleaved");
    }
  }

}

