using UnityEngine;

public class GGMLTest : MonoBehaviour {
  public string Path;
  public LlamaConfig Config = null;

  void Start() {
    var metadata = GGMLLoader.LoadMetadata(Path);
    Config = GGMLLoader.CreateConfig(metadata);
    Debug.Log($"GGML model {Path}:\n{Config}");
  }
}