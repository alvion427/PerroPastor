using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Conversation : MonoBehaviour {
  public Llama Llama;
  [TextArea]
  public string Query = "";
  public float Temperature = 0.9f;
  public int RunOnStart = 0;
  
  public Tokenizer Tokenizer => Llama.Tokenizer;

  public Action<string> OnNewToken;
  public Action<string> OnSequenceComplete;


  internal ComputeBuffer _outputToken;
  internal PersistentState _persistentState;
  internal int _pos = 0;
  internal int _tokensToRun = 0;
  internal bool _sequenceComplete = false;
  internal List<int> _queryTokens = new List<int>();
  internal List<int> _resultTokens = new List<int>();

  void Start() {
    Llama.StartConversation(this);
  }

  private void OnDestroy() {
  }

  internal void Initialize() {
    _outputToken = new ComputeBuffer(1, sizeof(int));
    _persistentState = new PersistentState(Llama.Config, Llama.RuntimeQuantizationMode);

    if (RunOnStart > 0) {
      RunTokens(Query, RunOnStart);
    }
  }

  internal void Shutdown() {
    _outputToken.Dispose();
    _persistentState.Dispose();
  }

  public void RunTokens(string query, int numTokens) {
    _tokensToRun = numTokens;
    _sequenceComplete = false;
    _resultTokens = new List<int>();
    
    if (query.Length > 0) {
      var tokenizedQuery = Llama.Tokenizer.Tokenize(query);
      _queryTokens = tokenizedQuery.ToList();
      tokenizedQuery.Dispose();
    }

    // Put start of sequence token in last token buffer.
    _outputToken.SetData(new int[] { Llama.Tokenizer.SOS });
  }

  internal void ProduceToken(int token, bool isFinalToken) {
    _resultTokens.Add(token);
    string tokenString = Tokenizer.Detokenize(token);

    OnNewToken?.Invoke(tokenString);
    
    if (isFinalToken) {
      SequenceComplete();
    }
  }

  internal void SequenceComplete() {
    _sequenceComplete = true;
    _tokensToRun = 0;
    string fullSequence = Tokenizer.Detokenize(_resultTokens);
    Debug.Log("Sequence complete: " + fullSequence);
    OnSequenceComplete?.Invoke(fullSequence);
  }

 

}
