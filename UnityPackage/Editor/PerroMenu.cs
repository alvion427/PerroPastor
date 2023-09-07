using System.Diagnostics;
using UnityEngine;
using UnityEditor;
using System.IO;
using UnityEngine.Networking;
using Debug = UnityEngine.Debug;

public class PerroMenu {
  [MenuItem("Perro/Open Models Folder")]
  public static void OpenModelsFolder() {
    string modelsFolder = Path.Combine(Application.persistentDataPath, "Models");
    string readmePath = Path.Combine(modelsFolder, "readme.txt");
    if (!Directory.Exists(modelsFolder)) {
      Directory.CreateDirectory(modelsFolder);
    }
    if (!File.Exists(readmePath)) {
      string readme =
        "Save ggml model files (from HuggingFace, etc) into this folder to use them in Perror Pastor." +
        "Once you save the file, you have to update the ModelPath in the GGMLLoader component on the Llama" +
        "object.  You can just include the file name as long as they are in this folder.";
      File.WriteAllText(readmePath, readme);
    }

    EditorUtility.RevealInFinder(readmePath);
  }

  [MenuItem("Perro/Download Stories Model")]
  public static void DownloadStoriesModel() {
    DownloadFile(
      "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
      Path.Combine(Application.persistentDataPath, "Models/stories15M.bin"));
    DownloadFile(
      "https://github.com/alvion427/PerroPastor/raw/master/Data/tokenizer2.bin",
      Path.Combine(Application.persistentDataPath, "Models/tokenizer.bin"));
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

    checkProgress = () => {
      if (www.isDone) {
        EditorApplication.update -= checkProgress;

        // Ensure the directory exists
        Directory.CreateDirectory(Path.GetDirectoryName(savePath));

        if (www.result == UnityWebRequest.Result.ConnectionError ||
            www.result == UnityWebRequest.Result.ProtocolError) {
          Debug.LogError(www.error);
        }
        else {
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

[CustomEditor(typeof(ModelLoaderBase))]
public class ModelLoaderBaseEditor : Editor {
  public override void OnInspectorGUI() {
    DrawDefaultInspector(); // Draws the default inspector

    // Draw a divider
    EditorGUILayout.Space();
    EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
    
    ModelLoaderBase myScript = (ModelLoaderBase)target;

    // Button to open file browser
    if (GUILayout.Button("Select Model"))
    {
      string initialPath = Path.Combine(Application.persistentDataPath, "Models");
      string selectedFile = EditorUtility.OpenFilePanel("Select a Model File", initialPath, "bin");

      if (!string.IsNullOrEmpty(selectedFile)) {
        myScript.ModelsDir = Path.GetDirectoryName(selectedFile);
        myScript.ModelPath = Path.GetFileName(selectedFile);
        EditorUtility.SetDirty(myScript); // Mark the object as changed so Unity knows to save the new value
      }
    }
    if (GUILayout.Button("Open Models Folder")) {
      PerroMenu.OpenModelsFolder();
    }
  }
}

[CustomEditor(typeof(KarpathyLoader))]
public class KarpathyLoaderEditor : ModelLoaderBaseEditor {
}

[CustomEditor(typeof(GGMLLoader))]
public class GGMLLoaderEditor : ModelLoaderBaseEditor {
}
