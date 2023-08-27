using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

public abstract class ModelLoaderBase : MonoBehaviour {
  public string ModelPath;
  public Action<LlamaConfig, Weights, Tokenizer> OnLoaded;
  
  public Tokenizer Tokenizer { get; protected set; }

  public bool IsLoaded { get; private set; }
  private Task<(LlamaConfig, Weights, Tokenizer)> _task;
  
  public void RequestLoad(QuantizationModes weightQuantMode, QuantizationModes runtimeQuantMode) {
    Debug.Assert(!IsLoaded && _task == null);
    _task = LoadModelImpl(weightQuantMode, runtimeQuantMode);
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

  protected abstract Task<(LlamaConfig, Weights, Tokenizer)> LoadModelImpl(QuantizationModes weightQuantMode, QuantizationModes runtimeQuantMode);
}
