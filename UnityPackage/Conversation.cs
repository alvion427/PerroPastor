using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;

public class Conversation : MonoBehaviour {
  public Llama Llama;
  [TextArea]
  public string Prompt = "";
  public float Temperature = 0.9f;
  public int RunOnStart = 0;
  public string TerminationSequence = "";

  public bool UsePromptCache = true;
  
  public Tokenizer Tokenizer => Llama.Tokenizer;

  public Action<string> OnNewToken;
  public Action<string> OnSequenceComplete;

  internal ComputeBuffer _outputToken;
  internal PersistentState _persistentState;
  internal int _pos = 0;
  internal int _tokensToRun = 0;
  internal bool _sequenceComplete = false;
  internal List<int> _terminationTokens;
  internal List<int> _resultTokens = new List<int>();

  private int _initialPromptLength = 0;

  void Start() {
    Llama.StartConversation(this);
  }

  private void OnDestroy() {
    Shutdown();
  }

  internal void Initialize() {
    _outputToken = new ComputeBuffer(1, sizeof(int));
    _persistentState = new PersistentState(Llama.Config, Llama.RuntimeQuantizationMode);

    // Put start of sequence token in last token buffer.
    _outputToken.SetData(new int[] { Llama.Tokenizer.SOS });

    if (UsePromptCache) {
      LoadPromptFromCache();
    }
    
    if (!string.IsNullOrEmpty(TerminationSequence)) {
      var tokens = Tokenizer.Tokenize(TerminationSequence);
      _terminationTokens = tokens.ToList();
      tokens.Dispose();
    }

    if (RunOnStart > 0) {
      RunTokens(Prompt, RunOnStart);
      _initialPromptLength = _resultTokens.Count;
    }
  }

  internal void Shutdown() {
    _outputToken.Dispose();
    _persistentState.Dispose();
  }

  public void RunTokens(string prompt, int numTokens = int.MaxValue) {
    int tokensLeft = Mathf.Max(Llama.Config.seq_len - _resultTokens.Count, 0);
    _tokensToRun = Mathf.Min(numTokens, tokensLeft);
    if (_tokensToRun == 0) {
      return;
    }
    
    _sequenceComplete = false;
    
    if (prompt.Length > 0) {
      var tokenizedPrompt = Llama.Tokenizer.Tokenize(prompt);
      _resultTokens.AddRange(tokenizedPrompt);
      tokenizedPrompt.Dispose();
    }
  }

  internal void ProduceToken(int pos, int token, bool isFinalToken) {
    if (pos == _resultTokens.Count) {
      _resultTokens.Add(token);
    }
    string tokenString = Tokenizer.Detokenize(token);

    if (UsePromptCache && _initialPromptLength > 0 && pos + 1 == _initialPromptLength) {
      SavePromptInCache();
    }

    OnNewToken?.Invoke(tokenString);
    
    if (_resultTokens.Count > _initialPromptLength && _resultTokens.Count >= _terminationTokens.Count && 
        _resultTokens.TakeLast(_terminationTokens.Count).SequenceEqual(_terminationTokens)) {
      isFinalToken = true;
    }
    
    if (isFinalToken) {
      Llama.SequenceComplete(this);
    }
  }

  internal void SequenceComplete() {
    _sequenceComplete = true;
    _tokensToRun = 0;
    string fullSequence = Tokenizer.Detokenize(_resultTokens);
    Debug.Log("Sequence complete: " + fullSequence);
    OnSequenceComplete?.Invoke(fullSequence);
  }

  private string GetPromptCachePath() {
    string cleanedInput = string.Concat(Prompt.Split(Path.GetInvalidFileNameChars()));
    string prefix = cleanedInput.Substring(0, Math.Min(16, cleanedInput.Length));
    string hash = Prompt.GetHashCode().ToString("x8");
    return Path.Combine(Application.persistentDataPath, "prompt_cache", $"{prefix}_{hash}.bin");    
  }

  private void SavePromptInCache() {
    string path = GetPromptCachePath();
    Debug.Log("Saving prompt in prompt cache...\n" + path);
    Directory.CreateDirectory(Path.GetDirectoryName(path));
    using (FileStream fs = new FileStream(path, FileMode.Create))
    {
      BinaryFormatter formatter = new BinaryFormatter();
      formatter.Serialize(fs, Llama.Config);
      formatter.Serialize(fs, Llama.RuntimeQuantizationMode);
      formatter.Serialize(fs, Prompt);
      formatter.Serialize(fs, _resultTokens);

      int cacheSize = _resultTokens.Count * Llama.Config.dim;
      
      for (int l = 0; l < _persistentState.layers.Length; ++l) {
        if (Llama.RuntimeQuantizationMode == QuantizationModes.Float32) {
          float[] keys = new float[cacheSize];
          float[] values = new float[cacheSize];
          _persistentState.layers[l].key_cache.GetData(keys);
          _persistentState.layers[l].value_cache.GetData(values);
          formatter.Serialize(fs, keys);
          formatter.Serialize(fs, values);
        }
        else if (Llama.RuntimeQuantizationMode == QuantizationModes.Float16) {
          half[] keys = new half[cacheSize];
          half[] values = new half[cacheSize];
          _persistentState.layers[l].key_cache.GetData(keys);
          _persistentState.layers[l].value_cache.GetData(values);
          formatter.Serialize(fs, keys);
          formatter.Serialize(fs, values);
        }
      }
    }
    Debug.Log("Done.");
  }
  
  private bool LoadPromptFromCache() {
    string path = GetPromptCachePath();
    if (!File.Exists(path)) {
      return false;
    }
    
    using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read))
    {
      BinaryFormatter formatter = new BinaryFormatter();
        
      LlamaConfig deserializedConfig = (LlamaConfig)formatter.Deserialize(fs);
      QuantizationModes runtimeMode = (QuantizationModes)formatter.Deserialize(fs);
      if (!Llama.Config.Equals(deserializedConfig) || Llama.RuntimeQuantizationMode != runtimeMode)
      {
        return false;
      }

      Prompt = (string)formatter.Deserialize(fs);
      var promptTokens = (List<int>)formatter.Deserialize(fs);
      _initialPromptLength = promptTokens.Count;

      for (int l = 0; l < _persistentState.layers.Length; ++l)
      {
        if (Llama.RuntimeQuantizationMode == QuantizationModes.Float32)
        {
          float[] keys = (float[])formatter.Deserialize(fs);
          float[] values = (float[])formatter.Deserialize(fs);
          _persistentState.layers[l].key_cache.SetData(keys);
          _persistentState.layers[l].value_cache.SetData(values);
        }
        else if (Llama.RuntimeQuantizationMode == QuantizationModes.Float16)
        {
          half[] keys = (half[])formatter.Deserialize(fs);
          half[] values = (half[])formatter.Deserialize(fs);
          _persistentState.layers[l].key_cache.SetData(keys);
          _persistentState.layers[l].value_cache.SetData(values);
        }
      }
      
      _pos = _initialPromptLength;
      _outputToken.SetData(new int[] {promptTokens[^1]});

      Debug.Log("Loaded prompt from cache: " + Prompt);
      OnNewToken?.Invoke(Prompt);
    }
    return true;
  }

}
