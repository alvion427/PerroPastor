using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

public abstract class ModelLoaderBase : MonoBehaviour {
  public string ModelPath;
  public string ModelsDir;
  public Action<LlamaConfig, WeightsGpu, Tokenizer> OnLoaded;
  
  public Tokenizer Tokenizer { get; protected set; }

  public bool IsLoaded { get; private set; }
  private Task<(LlamaConfig, WeightsGpu, Tokenizer)> _task;
  
  public void RequestLoad() {
    Debug.Assert(!IsLoaded && _task == null);
    _task = LoadModelImpl();
  }

  public string GetFullModelPath() {
    if (Path.IsPathRooted(ModelPath)) {
      return ModelPath;
    }
    else {
      if (string.IsNullOrEmpty(ModelsDir)) {
        ModelsDir = Path.Combine(Application.persistentDataPath, "Models");
      }
      return Path.Combine(ModelsDir, ModelPath);
    }
  }

  void Update() {
    // TODO: Just make this a continuation on Unity main thread when we add that.
    if (_task != null && _task.IsCompleted) {
      IsLoaded = true;
      (var config, var weights, var tokenizer) = _task.Result;
      Tokenizer = tokenizer;
      OnLoaded?.Invoke(config, weights, tokenizer);
      _task = null;
    }
  }

  protected abstract Task<(LlamaConfig, WeightsGpu, Tokenizer)> LoadModelImpl();
}
