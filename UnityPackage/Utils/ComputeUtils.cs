using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

public static class ComputeUtils {
  private static ComputeShader _computeShader;
  private static int _quantizeKernel = -1;
  private static int _dequantizeKernel = -1;
  private static int _blitKernel = -1;
  private static int _findRangeKernel = -1;

  // This is needed to avoid domain reloading between projects
  public static void Reset() {
    _computeShader = null;
    _quantizeKernel = -1;
  }
  
  public static int ElementCount<T>(this ComputeBuffer buffer) {
    return (buffer.count * buffer.stride) / Marshal.SizeOf<T>();
  }

  public static int ElementCount(this ComputeBuffer buffer, QuantizationModes quantMode) {
    return (buffer.count * buffer.stride) / quantMode.ElementSize();
  }

  public static int GetVectorizedLength(int rawLength) {
    //Debug.Assert(rawLength % 4 == 0);
    if (rawLength % 4 != 0) {
      Debug.LogError("Raw length is not a multiple of 4");
    }
    return rawLength / 4;
  }
  
  public static ComputeBuffer CreateVectorizedBuffer(int size, QuantizationModes mode) {
    Debug.Assert(size % 4 == 0);
    return new ComputeBuffer(size / 4, mode.ElementSize() * 4);
  }

  public static ComputeBuffer CreateBlockBuffer(int size, QuantizationModes mode) {
    Debug.Assert(size % mode.BlockSize() == 0);
    return new ComputeBuffer(size / mode.BlockSize(), mode.BlockSizeBytes());
  }

  public static void SetBuffer(this ComputeShader shader, int kernel, string name, GpuTensor tensor) {
    QuantizationUtil.EnableQuantizationKeywords(shader, tensor.Mode, "WEIGHT");
    shader.SetBuffer(kernel, name, tensor.Buffer);
  }

  public static void Quantize(QuantizationModes sourceMode, QuantizationModes destMode,
    ComputeBuffer inputBuffer, ComputeBuffer outputBuffer) {
    
    _loadShader(sourceMode, destMode);

    int count = inputBuffer.ElementCount(sourceMode);
    Debug.Assert(count % destMode.BlockSize() == 0);
    int numBlocks = count / destMode.BlockSize();

    // We only dispatch a maximum of 1024 groups with 1024 threads each loop
    const int dispatchSize = 1024 * 1024;
    int numDispatches = Mathf.CeilToInt(numBlocks / (float)dispatchSize);

    for (int d = 0; d < numDispatches; ++d) {
      int offset = d * dispatchSize;
      int batchNumBlocks = Mathf.Min(numBlocks - offset, dispatchSize);
      _computeShader.SetBuffer(_quantizeKernel, "quantize_input", inputBuffer);
      _computeShader.SetBuffer(_quantizeKernel, "quantize_output", outputBuffer);
      _computeShader.SetInt("quantize_numBlocks", batchNumBlocks);
      _computeShader.SetInt("quantize_offset", offset);

      int threadGroupsX = Mathf.CeilToInt(batchNumBlocks / 1024.0f);
      _computeShader.Dispatch(_quantizeKernel, threadGroupsX, 1, 1);
    }

  }

  public static void Dequantize(QuantizationModes sourceMode, QuantizationModes destMode,
    ComputeBuffer inputBuffer, ComputeBuffer outputBuffer) {
    
    _loadShader(sourceMode, destMode);

    int count = outputBuffer.ElementCount(destMode);
    Debug.Assert(count % sourceMode.BlockSize() == 0);
    int numBlocks = count / sourceMode.BlockSize();

    // We only dispatch a maximum of 1024 groups with 1024 threads each loop
    const int dispatchSize = 1024 * 1024;
    int numDispatches = Mathf.CeilToInt(numBlocks / (float)dispatchSize);

    for (int d = 0; d < numDispatches; ++d) {
      int offset = d * dispatchSize;
      int batchNumBlocks = Mathf.Min(numBlocks - offset, dispatchSize);
      _computeShader.SetBuffer(_dequantizeKernel, "dequantize_input", inputBuffer);
      _computeShader.SetBuffer(_dequantizeKernel, "dequantize_output", outputBuffer);
      _computeShader.SetInt("dequantize_numBlocks", batchNumBlocks);
      _computeShader.SetInt("dequantize_offset", offset);

      int threadGroupsX = Mathf.CeilToInt(batchNumBlocks / 1024.0f);
      _computeShader.Dispatch(_dequantizeKernel, threadGroupsX, 1, 1);
    }

  }

  public static void Quantize(QuantizationModes sourceMode, QuantizationModes destMode,
    NativeArray<byte> inputData, ComputeBuffer outputBuffer) {
    if (sourceMode == destMode) {
      outputBuffer.SetData(inputData);
      return;
    }

    int count = inputData.Length / sourceMode.ElementSize();
    Debug.Assert(count % destMode.BlockSize() == 0);

    ComputeBuffer stagingBuffer = CreateVectorizedBuffer(count, sourceMode);
    stagingBuffer.SetData(inputData);
    
    Quantize(sourceMode, destMode, stagingBuffer, outputBuffer);
    
    stagingBuffer.Dispose();
  }

    public static void FindRange(QuantizationModes sourceMode, ComputeBuffer inputBuffer, ComputeBuffer resultBuffer) {
      _loadShader(sourceMode, QuantizationModes.Float32);

      int blockCount = inputBuffer.count;
      _computeShader.SetBuffer(_findRangeKernel, "findrange_input", inputBuffer);
      _computeShader.SetBuffer(_findRangeKernel, "findrange_output", resultBuffer);
      _computeShader.SetInt("findrange_blockCount", blockCount);
      int threadGroupsX = Mathf.CeilToInt(blockCount / 1024.0f);
      _computeShader.Dispatch(_findRangeKernel, threadGroupsX, 1, 1);
    }

    public static RenderTexture BlitToTexture(QuantizationModes sourceMode, ComputeBuffer buffer, int width, int height) {
      RenderTexture result = new RenderTexture(width, height, 0, RenderTextureFormat.R8);
      result.enableRandomWrite = true;
      result.Create();
      BlitToTexture(sourceMode, buffer, width, height, result);
      return result;
    }

    public static void BlitToTexture(QuantizationModes sourceMode, ComputeBuffer buffer, int width, int height, RenderTexture result) {
      _loadShader(sourceMode, QuantizationModes.Q8_0);
      
      ComputeBuffer minmaxBuffer = new ComputeBuffer(2, sizeof(int));
      minmaxBuffer.SetData(new int[] { 100 * 256 * 256 * 256, -100 * 256 * 256 * 256 });
      FindRange(sourceMode, buffer, minmaxBuffer);
      
      int[] minmax = new int[2];
      minmaxBuffer.GetData(minmax);
      float min = minmax[0] / (256.0f * 256.0f);
      float max = minmax[1] / (256.0f * 256.0f);

      int blockWidth = width / sourceMode.BlockSize();
      
      _computeShader.SetBuffer(_blitKernel, "blit_input", buffer);
      _computeShader.SetTexture(_blitKernel, "blit_output", result);
      _computeShader.SetBuffer(_blitKernel, "blit_minmax", minmaxBuffer);
      _computeShader.SetInt("blit_blockWidth", blockWidth);
      _computeShader.SetFloat("blit_height", height);
      
      int threadGroupsX = Mathf.CeilToInt(blockWidth / 32.0f);
      int threadGroupsY = Mathf.CeilToInt(height / 32.0f);
      _computeShader.Dispatch(_blitKernel, threadGroupsX, threadGroupsY, 1);
      
      minmaxBuffer.Dispose();
    }

  private static void _loadShader(QuantizationModes sourceMode, QuantizationModes destMode) {
    if (_computeShader == null) {
      _computeShader = Resources.Load<ComputeShader>("ComputeUtils");
      _quantizeKernel = _computeShader.FindKernel("Quantize");
      _dequantizeKernel = _computeShader.FindKernel("Dequantize");
      _blitKernel = _computeShader.FindKernel("BlitToTexture");
      _findRangeKernel = _computeShader.FindKernel("FindRange");
    }
    QuantizationUtil.EnableQuantizationKeywords(_computeShader, sourceMode, "SOURCE");
    QuantizationUtil.EnableQuantizationKeywords(_computeShader, destMode, "DEST");
  }

}

