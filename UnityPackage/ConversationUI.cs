using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ConversationUI : MonoBehaviour {
    public Conversation Conversation;
    public TMPro.TMP_Text TextOutput;
    public TMPro.TMP_Text TextInput;
    public Button SubmitButton;
    
    private void OnEnable() {
        Conversation.OnNewToken += OnNewToken;
        SubmitButton.onClick.AddListener(OnSubmit);
    }

    private void OnDisable() {
        Conversation.OnNewToken -= OnNewToken;
    }

    void OnNewToken(string token) {
        TextOutput.text += token;
    }

    void OnSubmit() {
        string prompt = " " + TextInput.text;
        Conversation.RunTokens(prompt, Conversation.Llama.Config.seq_len);
        TextInput.text = "";
    }
}
