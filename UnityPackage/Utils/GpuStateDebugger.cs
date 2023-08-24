using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Directory = UnityEngine.Windows.Directory;

class GpuStateDebugger {
    public enum Modes {
        Save,
        Check,
    }

    public Modes Mode;
    public string Path; 

    public GpuStateDebugger(Modes mode, string recordingName) {
        Mode = mode;
        Path = System.IO.Path.Combine(Application.persistentDataPath, "state_recordings", recordingName);
        
        if (mode == Modes.Check) {
            LoadStates();
        }
    }
    
    public void TraceFinished() {
        if (Mode == Modes.Save) {
            SaveStates();
        }
    }

    private void LoadStates() {
        foreach (string file in System.IO.Directory.EnumerateFiles(Path)) {
            string name = System.IO.Path.GetFileNameWithoutExtension(file);
            int fileLen = (int)new System.IO.FileInfo(file).Length;
            int count = fileLen / sizeof(float);
            var reader = new BinaryReader(System.IO.File.OpenRead(file));
            float[] data = new float[count];
            for (int i = 0; i < count; i++) {
                data[i] = reader.ReadSingle();
            }
            SavedStates[name] = data;
        }
    }

    public void SaveStates() {
        Debug.Log("Saving trace states...");
        Directory.CreateDirectory(Path);
        foreach (var state in SavedStates) {
            string file = System.IO.Path.Combine(Path, state.Key + ".bin");
            var writer = new BinaryWriter(System.IO.File.OpenWrite(file));
            foreach (float f in state.Value) {
                writer.Write(f);
            }
            writer.Close();
        }
        Debug.Log("Done!");
    }

    public void ProcessState(string name, ComputeBuffer buffer, float warningTolerance = 0.001f) {
        float[] data = new float[buffer.ElementCount<float>()];
        buffer.GetData(data);

        if (Mode == Modes.Save) {
            Debug.Assert(!SavedStates.ContainsKey(name));
            SavedStates[name] = data;
            Debug.Log("Added state " + name);
        }
        else {
            float totalError = 0;
            float maxError = 0;
            for (int i = 0; i < data.Length; ++i) {
                float error = Mathf.Abs(data[i] - SavedStates[name][i]);
                totalError += error;
                maxError = Mathf.Max(maxError, error);
            }
            float avgError = totalError / data.Length;

            if (avgError > warningTolerance) {
                Debug.LogWarning($"EXCESSIVE ERROR {name} - Avg: {avgError}  Max: {maxError}");
            }
            else {
                Debug.Log($"State {name} passed - Avg: {avgError}  Max: {maxError}");
            }
        }
    }
    
    public Dictionary<string, float[]> SavedStates = new Dictionary<string, float[]>();
}

