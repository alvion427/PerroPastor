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

  public NativeArray<int> Tokenize(string text, bool sos) {
    List<int> tokens = new List<int>();

    if (sos) {
      tokens.Add(SOS);
    }

    for (int i = 0; i < text.Length; i++) {
      // Replace <sos> and <eos> with appropriate tokens and skip forward
      if (text.IndexOf("<sos>", i) == i)
      {
        tokens.Add(SOS);
        i += "<sos>".Length - 1;
        continue;
      }
      else if (text.IndexOf("<eos>", i) == i)
      {
        tokens.Add(EOS);
        i += "<eos>".Length - 1;
        continue;
      }

      tokens.Add(_textToToken[text[i].ToString()]);
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