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

    _loadShader(mode);

    // Sigh, kind of annoying that we have to copy the data once AGAIN to a staging buffer.  Probably the best way to
    // avoid this is to make our own serialization format and allow us to directly serialize quantized models.
    ComputeBuffer stagingBuffer = new ComputeBuffer(data.Length, sizeof(float));
    stagingBuffer.SetData(data);

    // We only dispatch a maximum of 1024 groups with 1024 threads each loop
    const int dispatchSize = 1024 * 1024;
    int numDispatches = Mathf.CeilToInt(vecLen / (float)dispatchSize);

    for (int d = 0; d < numDispatches; ++d) {
      int offset = d * dispatchSize;
      int batchVecLen = Mathf.Min(vecLen - offset, dispatchSize);
      _computeShader.SetBuffer(_setDataQuantizedKernel, "setquant_input", stagingBuffer);
      _computeShader.SetBuffer(_setDataQuantizedKernel, "setquant_output", outputBuffer);
      _computeShader.SetInt("setquant_veclen", batchVecLen);
      _computeShader.SetInt("setquant_offset", offset);

      int threadGroupsX = Mathf.CeilToInt(vecLen / 1024.0f);
      _computeShader.Dispatch(_setDataQuantizedKernel, threadGroupsX, 1, 1);
    }
    
    
    stagingBuffer.Dispose();
  }

  public static void SetQuantizedDataInterleaved(QuantizationModes mode, ComputeBuffer outputBuffer, NativeArray<float> dataA, NativeArray<float> dataB) {
    _loadShader(mode);

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

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _computeShader.Dispatch(_setDataQuantizedInterleavedKernel, threadGroupsX, 1, 1);
    stagingBufferA.Dispose();
    stagingBufferB.Dispose();
  }

  private static void _loadShader(QuantizationModes mode) {
    if (_computeShader == null) {
      _computeShader = Resources.Load<ComputeShader>("ComputeUtils");
      QuantizationUtil.EnableQuantizationKeywords(_computeShader, mode, QuantizationModes.Float32);
      _setDataQuantizedKernel = _computeShader.FindKernel("SetQuantizedData");
      _setDataQuantizedInterleavedKernel = _computeShader.FindKernel("SetQuantizedDataInterleaved");
    }
  }

}

