using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LlamaTextSetter : MonoBehaviour {
    public Llama Llama;
    public TMPro.TMP_Text Text;
    
    private void OnEnable() {
        Llama.OnNewToken += OnNewToken;
    }

    private void OnDisable() {
        Llama.OnNewToken -= OnNewToken;
    }

    void OnNewToken(string token) {
        Text.text += token;
    }
}
