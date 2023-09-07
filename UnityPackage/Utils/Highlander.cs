using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Highlander : MonoBehaviour
{
    void Awake() {
        // THERE CAN BE ONLY ONE

        bool foundOneActive = false;
        foreach (var llama in transform.GetComponentsInChildren<Llama>()) {
            if (!File.Exists(llama.GetComponent<ModelLoaderBase>().GetFullModelPath())) {
                llama.gameObject.SetActive(false);
            }
            else if (foundOneActive) {
                llama.gameObject.SetActive(false);
            }
            else if (llama.gameObject.activeSelf) {
                foundOneActive = true;
            }
        }
    }
}
