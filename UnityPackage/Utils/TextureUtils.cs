using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using System.IO;
using UnityEngine.Experimental.Rendering;

public static class TextureUtils {
  
  private static ComputeShader _computeShader;
  private static int _setTextureKernel = -1;
  
  public static void SetTextureData(RenderTexture tex, NativeArray<float> data) {
    _loadShader();

    tex.enableRandomWrite = true;

    ComputeBuffer stagingBuffer = new ComputeBuffer(data.Length, sizeof(float));
    stagingBuffer.SetData(data);
    _computeShader.SetTexture(_setTextureKernel, "settex_Texture", tex);
    _computeShader.SetBuffer(_setTextureKernel, "settex_StagingBuffer", stagingBuffer);

    _computeShader.Dispatch(_setTextureKernel, tex.width / 8, tex.height / 8, 1);
    stagingBuffer.Dispose();
    
    Texture2D checkTexture = tex.ToTexture2D();
    NativeArray<float> checkData = checkTexture.GetRawTextureData<float>();
    for (int i = 0; i < data.Length; ++i) {
      float source = data[i];
      float check = checkData[i];
      float diff = Mathf.Abs(check - (half)source);
      if (diff > 0.002f) {
        Debug.LogError($"SetTextureData doesn't match at {i} {check} != {source}");
        break;
      }
    }

    Object.Destroy(checkTexture);
  }

  public static Texture2D CreateMatchingTexture(Texture texture) {
    return new Texture2D(texture.width, texture.height, texture.graphicsFormat, TextureCreationFlags.None);
  }

  public static Texture2D ToTexture2D(this RenderTexture renderTexture) {
    Texture2D result = CreateMatchingTexture(renderTexture);
    ToTexture2D(renderTexture, result);
    return result;
  }

  public static void ToTexture2D(RenderTexture renderTexture, Texture2D dest) {
    RenderTexture old = RenderTexture.active;
    RenderTexture.active = renderTexture;
    dest.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
    dest.Apply();
    var data = dest.GetRawTextureData<float>();
    RenderTexture.active = old;
  }

  public static void WriteToFile(this RenderTexture texture, string path) {
    var texture2d = texture.ToTexture2D();
    WriteToFile(texture2d, path);
    Object.Destroy(texture2d);
  }

  public static void WriteToFile(this Texture2D texture, string path) {
    byte[] bytes = texture.EncodeToPNG();
    if (!Path.IsPathRooted(path)) {
      path = Path.Combine(Application.persistentDataPath, $"{path}");
    }
    File.WriteAllBytes(path, bytes);
  }

  private static void _loadShader() {
    if (_computeShader == null) {
      _computeShader = Resources.Load<ComputeShader>("TextureUtils");
      _setTextureKernel = _computeShader.FindKernel("SetTexture");
    }
  }
}
