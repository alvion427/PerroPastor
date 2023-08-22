using UnityEngine;
using UnityEditor;
using System.IO;
using UnityEngine.Networking;

public class PerroMenu
{
  [MenuItem("Perro/Download Sample Model")]
  public static void DownloadModel()
  {
    DownloadFile(
      "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
      Path.Combine(Application.streamingAssetsPath, "Models/stories15M.bin"));
    DownloadFile(
      "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
      Path.Combine(Application.streamingAssetsPath, "Models/tokenizer.bin"));
  }

  public static void DownloadFile(string url, string savePath) {
    if (File.Exists(savePath)) {
      Debug.Log($"Skipping {Path.GetFileName(savePath)}, it already exists");
      return;
    }
    
    Debug.Log($"Downloading {url} to {savePath}");
    
    // Start the download
    UnityWebRequest www = UnityWebRequest.Get(url);
    www.SendWebRequest();

    EditorApplication.CallbackFunction checkProgress = null;
    
    checkProgress = () =>
    {
      if (www.isDone)
      {
        EditorApplication.update -= checkProgress;

        // Ensure the directory exists
        Directory.CreateDirectory(Path.GetDirectoryName(savePath));

        if (www.result == UnityWebRequest.Result.ConnectionError || www.result == UnityWebRequest.Result.ProtocolError)
        {
          Debug.LogError(www.error);
        }
        else
        {
          File.WriteAllBytes(savePath, www.downloadHandler.data);
          Debug.Log($"Finished downloading {Path.GetFileName(savePath)}");
        }

        www.Dispose();
        www = null;
      }
    };

    // Register the update callback
    EditorApplication.update += checkProgress;
  }
}