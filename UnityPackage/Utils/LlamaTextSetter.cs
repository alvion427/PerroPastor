using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LlamaTextSetter : MonoBehaviour {
    public Conversation Conversation;
    public TMPro.TMP_Text Text;
    
    private void OnEnable() {
        Conversation.OnNewToken += OnNewToken;
    }

    private void OnDisable() {
        Conversation.OnNewToken -= OnNewToken;
    }

    void OnNewToken(string token) {
        Text.text += token;
    }
}
