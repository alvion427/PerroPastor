using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public static class WeightTextureExport
{
  public static void ExportWeights(LlamaConfig config, WeightsGpu weights, string folderPath) {
    if (!Directory.Exists(folderPath)) {
      Directory.CreateDirectory(folderPath);
    }

    string[] weightTypes = new string[] {
      "wq", "wk", "wv", "wo", "w1", "w2", "w3"
    };
  
    foreach (string weightName in weightTypes) {
      Debug.Log("Exporting " + weightName + "...");
      var field = typeof(LayerWeightsGPU).GetField(weightName);
      for (int l = 0; l < config.n_layers; ++l) {
        GpuTensor weightsTensor = field.GetValue(weights.layerWeights[l]) as GpuTensor;
        RenderTexture wqTexture = weightsTensor.ConvertToTexture();
        wqTexture.WriteToFile(Path.Combine(folderPath, $"{weightName}_{l}.png"));
      }
    }
    
    Debug.Log("Export complete!");
  }

  public static RenderTexture CreateTextureAtlas(RenderTexture[] textures) {
    int w = textures[0].width;
    int h = textures[0].height;

    int totalArea = w * h * textures.Length;
    float squareSide = Mathf.Sqrt(totalArea);
    int gridWidth = Mathf.RoundToInt(squareSide / w);
    int gridHeight = (int)((textures.Length + gridWidth - 1) / (float)gridWidth); 
    
    int atlasWidth = gridWidth * w;
    int atlasHeight = gridHeight * w;

    RenderTexture atlas = new RenderTexture(atlasWidth, atlasHeight, 0);
    atlas.enableRandomWrite = true;
    atlas.Create();

    int xOffset = 0;
    int yOffset = 0;

    foreach (var texture in textures)
    {
      Graphics.CopyTexture(texture, 0, 0, 0, 0, w, h, atlas, 0, 0, xOffset, yOffset);

      xOffset += w;

      if (xOffset >= atlasWidth)
      {
        xOffset = 0;
        yOffset += h;
      }
    }

    return atlas;
  }

  // Use the previous ClosestSquareDimensions function
  private static (int, int) ClosestSquareDimensions(int number)
  {
    int sqrt = (int)Mathf.Sqrt(number);

    for (int i = sqrt; i > 0; i--)
    {
      if (number % i == 0)
      {
        return (i, number / i);
      }
    }

    throw new ArgumentException("Invalid input");
  }
}
