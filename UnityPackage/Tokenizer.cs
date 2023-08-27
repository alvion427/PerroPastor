using System;
using System.Collections.Generic;
using Unity.Collections;

public class Tokenizer {
  public int VocabSize { get; private set; }

  public int SOS { get; private set; }
  public int EOS { get; private set; }

  private Dictionary<string, int> _textToToken;
  private string[] _tokenToText;
  private float[] _tokenToScore;

  public Tokenizer(
    Dictionary<string, int> textToToken,
    string[] tokenToText,
    float[] tokenToScore,
    int vocabSize,
    int sos,
    int eos) {

    _textToToken = textToToken;
    _tokenToText = tokenToText;
    _tokenToScore = tokenToScore;
    VocabSize = vocabSize;
    SOS = sos;
    EOS = eos;
  }

  public string Detokenize(int token) {
    return _tokenToText[token];
  }
  
  public string Detokenize(List<int> tokens) {
    return string.Join("", tokens.ConvertAll<string>(x => Detokenize(x)));    
  }

  public NativeArray<int> Tokenize(string text) {
    List<int> tokens = new List<int>();

    foreach (char c in text) {
      if (_textToToken.TryGetValue(c.ToString(), out int id)) {
        tokens.Add(id);
      }
      else {
        throw new Exception($"Character '{c}' not found in vocabulary");
      }
    }

    while (true) {
      float bestScore = float.MinValue;
      int bestId = -1;
      int bestIdx = -1;

      for (int i = 0; i < (tokens.Count - 1); i++) {
        string pair = _tokenToText[tokens[i]] + _tokenToText[tokens[i + 1]];
        if (_textToToken.TryGetValue(pair, out int id) && _tokenToScore[id] > bestScore) {
          bestScore = _tokenToScore[id];
          bestId = id;
          bestIdx = i;
        }
      }

      if (bestIdx == -1)
        break;

      tokens[bestIdx] = bestId;
      tokens.RemoveAt(bestIdx + 1);
    }

    return new NativeArray<int>(tokens.ToArray(), Allocator.Persistent);
  }
}