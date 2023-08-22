using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Collections;
using UnityEngine;

public class Tokenizer : MonoBehaviour {
  public string VocabPath = "";
  public int VocabSize { get; private set; }

  public int SOS = 1;
  public int EOS = 2;

  private Dictionary<string, int> vocabStringToInt;
  private string[] vocabIntToString;
  private float[] vocabScores;
  private uint maxTokenLength;

  public string Detokenize(int token) {
    return vocabIntToString[token];
  }

  public NativeArray<int> Tokenize(string text) {
    List<int> tokens = new List<int>();

    foreach (char c in text) {
      if (vocabStringToInt.TryGetValue(c.ToString(), out int id)) {
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
        string pair = vocabIntToString[tokens[i]] + vocabIntToString[tokens[i + 1]];
        if (vocabStringToInt.TryGetValue(pair, out int id) && vocabScores[id] > bestScore) {
          bestScore = vocabScores[id];
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

  public void LoadTokenizer(int vocabSize) {
    VocabSize = vocabSize;

    vocabStringToInt = new Dictionary<string, int>();

    string fullPath = VocabPath;
    if (!File.Exists(fullPath)) {
      fullPath = Path.Combine(Application.streamingAssetsPath, "models", fullPath);
    }

    using (BinaryReader reader = new BinaryReader(File.Open(fullPath, FileMode.Open))) {
      maxTokenLength = reader.ReadUInt32();
      vocabScores = new float[VocabSize];
      vocabIntToString = new string[VocabSize];

      for (int i = 0; i < vocabSize; i++) {
        vocabScores[i] = reader.ReadSingle();
        int len = reader.ReadInt32();
        var bytes = reader.ReadBytes(len);
        string vocabEntry = Encoding.UTF8.GetString(bytes);
        vocabStringToInt[vocabEntry] = i;
        vocabIntToString[i] = vocabEntry;

        if (vocabEntry.Trim() == "<s>") {
          SOS = i;
        }
        else if (vocabEntry.Trim() == "</s>") {
          EOS = i;
        }
      }
    }

    Debug.Log($"Loaded {vocabIntToString} tokens from {VocabPath}");
  }
}