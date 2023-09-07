using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.UI;

public class ConversationUI : MonoBehaviour {
    public Conversation Conversation;
    public TMPro.TMP_Text TextOutput;
    public TMPro.TMP_InputField TextInput;
    public Button SubmitButton;
    
    private void OnEnable() {
        Conversation.OnNewToken += OnNewToken;
        SubmitButton.onClick.AddListener(OnSubmit);
        TextInput.onEndEdit.AddListener(OnPressEnter);
    }

    private void OnDisable() {
        Conversation.OnNewToken -= OnNewToken;
        SubmitButton.onClick.RemoveListener(OnSubmit);
        TextInput.onEndEdit.RemoveListener(OnPressEnter);
    }

    void OnNewToken(string token) {
        TextOutput.text += token;
    }

    void OnPressEnter(string prompt) {
        OnSubmit();
    }
    
    void OnSubmit() {
        TextInput.ActivateInputField();
        string prompt = Regex.Unescape(Conversation.PromptPrefix + TextInput.text + Conversation.PromptPostfix);
        TextInput.text = "";
        Conversation.RunTokens(prompt, Conversation.Llama.Config.seq_len);
    }
}
