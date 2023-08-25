using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

public class Llama : MonoBehaviour {
  public string Query = "";
  public float Temperature = 0.9f;
  public int RandomSeed = -1; // -1 means use time
  public int RunOnStart = 0;

  public QuantizationModes QuantizationMode = QuantizationModes.Float32;
  public string CheckpointPath;
  public Tokenizer Tokenizer;
  public ComputeShader llamaShader;

  public int MaxTokensPerFrame = 1;

  public Action<string> OnNewToken;
  public Action<string> OnSequenceComplete;

  private int _pos = 0;
  private int _tokensToRun = 0;
  private bool _sequenceComplete = false;
  public List<int> QueryTokens = new List<int>();
  public List<int> ResultTokens = new List<int>();

  public string SaveTrace;
  public string CheckTrace;

  private bool _isInitialized = false;
  private LlamaConfig _config;
  private Weights _weights;
  private WeightsGPU _weightsGPU;
  private RunState _runState;
  private PersistentState _persistentState;
  private System.Random _rng;
  private GpuStateDebugger _gpuStateDebugger;
  private LlamaKernels _kernels;
  
  public bool _Debug = false;

  void Start() {
    if (!string.IsNullOrEmpty(CheckTrace)) {
      _gpuStateDebugger = new GpuStateDebugger(GpuStateDebugger.Modes.Check, CheckTrace);
    }
    else if (!string.IsNullOrEmpty(SaveTrace)) {
      _gpuStateDebugger = new GpuStateDebugger(GpuStateDebugger.Modes.Save, SaveTrace);
    }

    if (Tokenizer == null) {
      Tokenizer = GetComponent<Tokenizer>();
    }

    _rng = new System.Random(RandomSeed == -1 ? (int)DateTime.Now.Ticks : RandomSeed);

    LoadShader();
    Initialize();

    if (RunOnStart > 0) {
      RunTokens(RunOnStart);
    }
  }

  private void OnDestroy() {
    if (_isInitialized) {
      Uninitialize();
    }
  }

  public void RunTokens(int numTokens) {
    _tokensToRun = numTokens;
    _sequenceComplete = false;
    ResultTokens = new List<int>();
    
    if (Query.Length > 0) {
      var tokenizedQuery = Tokenizer.Tokenize(Query);
      QueryTokens = tokenizedQuery.ToList();
      tokenizedQuery.Dispose();
    }

    // Put start of sequence token in last token buffer.
    _runState.outputToken.SetData(new int[] { Tokenizer.SOS });
  }

  void Update() {
    int tokensProcessedThisFrame = 0;
    while (!_sequenceComplete && _tokensToRun > 0 && tokensProcessedThisFrame < MaxTokensPerFrame) {
      ++tokensProcessedThisFrame;

      RunTransformer(_pos);

      if (_pos < QueryTokens.Count) {
        int queryToken = QueryTokens[_pos];
        _runState.outputToken.SetData(new int[] { queryToken });
        ProduceToken(queryToken);
      }
      else {
        if (Temperature == 0) {
          FindMaxIndex(_runState.logits, _runState.outputToken, _config.vocab_size);
        }
        else {
          ScaleBuffer(_runState.logits, 1.0f / Temperature, _config.vocab_size);
          Softmax(_runState.logits, 0, _config.vocab_size);
          SampleLogits(_runState.logits, (float)_rng.NextDouble());
        }
        
        bool isFinalToken = _tokensToRun == 1;

        AsyncGPUReadback.Request(_runState.outputToken, (request) => {
          if (_sequenceComplete) {
            return;
          }
          
          if (request.hasError) {
            Debug.LogError("Failed to readback output token buffer");
            enabled = false;
            return;
          }

          Debug.Assert(ResultTokens.Count < _config.seq_len);

          int token = request.GetData<int>()[0];
          if (token == Tokenizer.SOS || token == Tokenizer.EOS) {
            SequenceComplete();
            return;
          }

          ProduceToken(token);

          if (isFinalToken) {
            SequenceComplete();
          }
        });
      }

      --_tokensToRun;
      ++_pos;

      if (_tokensToRun == 0) {
        SequenceComplete();
      }
    }
  }

  private void ProduceToken(int token) {
    ResultTokens.Add(token);
    string tokenString = Tokenizer.Detokenize(token);
    if (_Debug) {
      Debug.Log($"Output token: {token} {tokenString}");
    }

    OnNewToken?.Invoke(tokenString);
  }

  private void SequenceComplete() {
    _gpuStateDebugger?.TraceFinished();
    _sequenceComplete = true;
    _tokensToRun = 0;
    string fullSequence =
      string.Join("", ResultTokens.ConvertAll<string>(x => Tokenizer.Detokenize(x)));
    Debug.Log("Sequence complete: " + fullSequence);
    OnSequenceComplete?.Invoke(fullSequence);
  }

  private void LoadShader() {
    if (llamaShader == null)
      llamaShader = (ComputeShader)Resources.Load("Llama.compute");
    QuantizationUtil.EnableQuantizationKeywords(llamaShader, QuantizationMode);
    _kernels = new LlamaKernels(llamaShader);
  }

  public void RunTransformer(int pos) {
    if (!_isInitialized) {
      Initialize();
    }

    int dim = _config.dim;
    int hidden_dim = _config.hidden_dim;

    // The first step is to load the embedding for the current token.  This is just a simple lookup into the
    // embedding table of the last generated token (or the start of sequence token when we begin).
    LoadEmbedding(_runState.x, _runState.outputToken);
    _gpuStateDebugger?.ProcessState($"token{_pos}_load_embedding", _runState.x);

    // We process each layer more or less independently, and the results of each layer feed into the next layer.
    // The input for each layer is in runState.x, which initially is just the embedding of the input token.
    // At the end of each layer, we write the layers output (along with a residual connection to the input) back
    // into x to serve as the input for the next layer.
    for (int l = 0; l < _config.n_layers; l++) {
      // Unlike the original Transformer architecture, which scaled the output of attention by a LayerNorm, Llama
      // pre-scales the inputs to the layers using RMS (root-mean-squared) normalization.  
      // NOTE!  RMSNorm also sneaks in a scaling by a set of weights.  These weights are defined per-layer, and
      // allow each layer to emphasize different features in the input.
      RmsNorm(_runState.x, _weightsGPU.layerWeights[l].rms_att_weight, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_rmsnorm", _runState.xb);

      // QKV (query, key, value) matmuls for this position.
      // The normalized layer input is multiplied by (dim, dim) matrices wq, wk, and wv, the results of which
      // are stored in the (dim,) vectors q, k, and v respectively.  You can think of this at the query, key, and
      // value vectors for the current token.
      MatMul(_weightsGPU.layerWeights[l].wq, _runState.xb, _runState.q, dim, dim);
      MatMul(_weightsGPU.layerWeights[l].wk, _runState.xb, _runState.k, dim, dim);
      MatMul(_weightsGPU.layerWeights[l].wv, _runState.xb, _runState.v, dim, dim);

      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_q", _runState.q);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_k", _runState.k);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_v", _runState.v);

      // Apply RoPE rotation to the q and k vectors.
      // RoPE is an alternative to positional encoding that works by "rotating" the query and key vectors using
      // magic complex numbers!  It's a complicated subject with a lot of math, but for our purposes, the key
      // point is that this is where positional information is being fed into the network rather than the more 
      // traditional approach to absolute positioning.
      // NOTE: In the original Transformer paper, embeddings were added directly to the input vector, whereas in
      // Llama they are a permutation of the query and key vectors.  Applying the positional information inside
      // the attention mechanism rather than directly to the input makes intuitive sense, since the attention
      // mechanism is the only part of the network that takes the full sequence into account.
      Rope(_runState.q, _runState.k, pos, l);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_rope_q", _runState.q);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_rope_k", _runState.k);

      // As we process each token, we accumulate the key and value vectors into a key/value cache, which allows us
      // to reuse them for future tokens.
      Memcpy(_persistentState.layers[l].key_cache, _runState.k, 0, pos * dim, dim);
      Memcpy(_persistentState.layers[l].value_cache, _runState.v, 0, pos * dim, dim);

      // The results of attention are an accumulation in this buffer, so we zero it out for all heads before
      // computing attention 
      ClearBuffer(_runState.xbFixed);

      // Multi-head attention. Iterate over each head and compute it's attention
      for (int h = 0; h < _config.n_heads; h++) {
        // First we compute the raw attention scores for this head, stored in vector runState.att (seq_len,).
        // To do this, we simply take the query vector for the current token (stored in runState.q) and then for
        // each previous token in the sequence we dot product runState.q with keyCache.v[pos] and store it in
        // runState.att[pos].
        ComputeAttention(_runState.q, _persistentState.layers[l].key_cache, _runState.att, h, pos);

        // Normalize attension scores using softamx.
        // NOTE: This is currently implemented serially in a single gpu thread, which is obviously very bad.
        Softmax(_runState.att, h * _config.seq_len, pos + 1);

        // We now use the normalized attention scores to scale the values (computed above) and accumulate the
        // results in the output vector.  Because of the softmax above, the attention scores sum to 1, so this
        // is just a "weighted sum" of all value vectors using the attention scores as weights.
        // You can think of it like this:
        //  - Queries are a way of the current token asking "Here is what I'm looking for".
        //  - Keys are a way of the previous tokens saying "Here is what I have".
        //  - Attention scores are essentially a way of determining which Keys align with the Queries.
        //  - Values are learned feature vectors for each token, and in this step we do a weighted sum of all
        //    values based on how much the current query aligned with the key associated with that value.
        // NOTE: The output vector is using fixed point to allow for atomic operations, converted to floats
        //   below.
        WeightedSum(_persistentState.layers[l].value_cache, _runState.att, _runState.xbFixed, h, pos);
      }

      // Convert all fixed point output vectors from attention into float
      FixedToFloat(_runState.xbFixed, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_attention", _runState.xb);

      // Final matmul to get the output of the attention.  This allows us to apply some additional learned weights
      // to the output of the attention.
      MatMul(_weightsGPU.layerWeights[l].wo, _runState.xb, _runState.xb2, dim, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_weighted_attention", _runState.xb2);

      // Residual connection back into x.  This allows us to combine many layers without losing signal during
      // backpropagation through many layers.
      Accumulate(_runState.x, _runState.xb2, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_accumulate_attention", _runState.x);

      // A final RMS norm of the outputs of the attention module.
      RmsNorm(_runState.x, _weightsGPU.layerWeights[l].rms_ffn_weight, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_attention_norm", _runState.xb);

      // For the feedforward network, we use a 2-layer MLP with an additional elementwise multiply to allow the
      // network to learn to amplify or diminish the outputs from the first layer (ie gating).  Also unlike
      // the original Transformer, we use SiLU (swish) as the non-linearity.
      // The FFN would be expressed in pytorch as: w2(F.silu(w1(x)) * w3(x))

      // Multiply w1(x) to expand (dim,) to (hidden_dim,)
      MatMul(_weightsGPU.layerWeights[l].w1, _runState.xb, _runState.hb, hidden_dim, dim);

      // Silu nonlinearity
      Silu(_runState.hb, hidden_dim);

      // Multiply w3(x) to calculate gating parameters (hidden_dim,)
      MatMul(_weightsGPU.layerWeights[l].w3, _runState.xb, _runState.hb2, hidden_dim, dim);

      // Elementwise multiply with w3(x) to apply gating (amplify or diminish results of w1(x))
      Multiply(_runState.hb, _runState.hb2, _runState.hb, hidden_dim);

      // Multiply result by w2 to compress back to (dim,)
      MatMul(_weightsGPU.layerWeights[l].w2, _runState.hb, _runState.xb, dim, hidden_dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_ffn", _runState.xb);

      // Again, apply a residual connection back to the original layer input to preserve the gradient.
      Accumulate(_runState.x, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{_pos}_layer{l}_accumulate_ffn", _runState.x);
    }

    // Final rmsnorm of all layer results
    RmsNorm(_runState.x, _weightsGPU.rms_final_weight, _runState.xb, dim);
    _gpuStateDebugger?.ProcessState($"token{_pos}_final_rmsnorm", _runState.xb);

    // Classify normalized results into logits with one big, fat matmul
    MatMul(_weightsGPU.GetWCLS(), _runState.xb, _runState.logits, _config.vocab_size, dim);
    _gpuStateDebugger?.ProcessState($"token{_pos}_classifier", _runState.logits);
  }

  private void Initialize() {
    _isInitialized = true;

    LoadWeights(CheckpointPath);
    Tokenizer.LoadTokenizer(_config.vocab_size);
    _weightsGPU = new WeightsGPU(_config);
    _weightsGPU.LoadWeights(_config, _weights);

    _runState = new RunState(_config);
    _persistentState = new PersistentState(_config);
  }

  private void Uninitialize() {
    // TODO: Wait for finishing all tasks?

    _persistentState.Dispose();
    _runState.Dispose();

    _weightsGPU.Dispose();
    _weights.Dispose();

    _isInitialized = false;
  }

  private void ClearBuffer(ComputeBuffer dest) {
    Profiler.BeginSample("ClearBuffer");

    int length = dest.count;
    llamaShader.SetBuffer(_kernels.clear, "clear_dest", dest);
    llamaShader.SetInt("clear_length", length);
    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    llamaShader.Dispatch(_kernels.clear, threadGroupsX, 1, 1);

    Profiler.EndSample();
  }

  private void Memcpy(ComputeBuffer copy_dest, ComputeBuffer copy_source, int sourceOffset, int destOffset,
    int length) {
    Profiler.BeginSample("Memcpy");

    int vecLen = ComputeUtils.GetVectorizedLength(length);
    sourceOffset = ComputeUtils.GetVectorizedLength(sourceOffset);
    destOffset = ComputeUtils.GetVectorizedLength(destOffset);

    // Set the buffers
    llamaShader.SetBuffer(_kernels.memcpy, "copy_dest", copy_dest);
    llamaShader.SetBuffer(_kernels.memcpy, "copy_source", copy_source);

    // Set the length
    llamaShader.SetInt("memcpy_source_offset", sourceOffset);
    llamaShader.SetInt("memcpy_dest_offset", destOffset);
    llamaShader.SetInt("memcpy_veclen", vecLen);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    llamaShader.Dispatch(_kernels.memcpy, threadGroupsX, 1, 1);

    Profiler.EndSample();


    if (_Debug) {
      float[] resultData = new float[copy_dest.ElementCount<float>()];
      copy_dest.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, destOffset, 8));
      Debug.Log(debugString);
    }
  }

  private void ScaleBuffer(ComputeBuffer buffer, float scale, int length) {
    Profiler.BeginSample("ScaleBuffer");

    int vecLen = ComputeUtils.GetVectorizedLength(length);

    llamaShader.SetBuffer(_kernels.scaleBuffer, "scalebuffer_buffer", buffer);
    llamaShader.SetInt("scalebuffer_veclen", vecLen);
    llamaShader.SetFloat("scalebuffer_scale", scale);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    llamaShader.Dispatch(_kernels.scaleBuffer, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[buffer.ElementCount<float>()];
      buffer.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log("ScaleBuffer: " + debugString);
    }
  }

  private void FixedToFloat(ComputeBuffer fixedBuffer, ComputeBuffer floatBuffer, int length) {
    Profiler.BeginSample("FixedToFloat");

    int vecLength = ComputeUtils.GetVectorizedLength(length);

    // Set the buffers
    llamaShader.SetBuffer(_kernels.fixedToFloat, "fixedtofloat_source", fixedBuffer);
    llamaShader.SetBuffer(_kernels.fixedToFloat, "fixedtofloat_dest", floatBuffer);

    // Set the length
    llamaShader.SetInt("fixedtofloat_length", vecLength);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(vecLength / 256.0f);
    llamaShader.Dispatch(_kernels.fixedToFloat, threadGroupsX, 1, 1);

    Profiler.EndSample();
  }

  private void LoadEmbedding(ComputeBuffer embedding, ComputeBuffer token) {
    Profiler.BeginSample("loadEmbedding");

    int veclen = ComputeUtils.GetVectorizedLength(_config.dim);

    // Set the buffers
    llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_token", token);
    llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_source", _weightsGPU.token_embedding_table);
    llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_dest", embedding);
    llamaShader.SetInt("loadembedding_veclen", veclen);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(veclen / 256.0f);
    llamaShader.Dispatch(_kernels.loadEmbedding, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] tokenData = new int[token.ElementCount<int>()];
      token.GetData(tokenData);

      NativeArray<float> correctEmbedding = new NativeArray<float>(_config.dim, Allocator.Temp);
      NativeArray<float>.Copy(_weights.token_embedding_table, tokenData[0] * _config.dim, correctEmbedding, 0,
        _config.dim);

      float[] resultData = new float[embedding.ElementCount<float>()];
      embedding.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void MatMul(ComputeBuffer matrixW, ComputeBuffer vectorX, ComputeBuffer vectorOut, int rows, int cols) {
    Profiler.BeginSample("Matmul");

    int colsVec = ComputeUtils.GetVectorizedLength(cols);

    // W (d,n) @ x (n,) -> xout (d,)
    llamaShader.SetBuffer(_kernels.matmul, "matmul_matrixW", matrixW);
    llamaShader.SetBuffer(_kernels.matmul, "matmul_vectorX", vectorX);
    llamaShader.SetBuffer(_kernels.matmul, "matmul_vectorOut", vectorOut);
    llamaShader.SetInt("matmul_rows", rows);
    llamaShader.SetInt("matmul_cols_vec", colsVec);

    int threadGroupsX = Mathf.CeilToInt(rows / 256.0f);
    llamaShader.Dispatch(_kernels.matmul, threadGroupsX, 1, 1);
    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[vectorOut.ElementCount<float>()];
      vectorOut.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void MatMulTex(RenderTexture matrixW, ComputeBuffer vectorX, ComputeBuffer vectorOut, int rows, int cols) {
    Profiler.BeginSample("MatmulTex");

    int kernel = _kernels.matmulTex;

    // W (rows,cols) @ x (rows,) -> xout (d,)
    llamaShader.SetTexture(kernel, "matmultex_matrixW", matrixW);
    llamaShader.SetTexture(kernel, "sampler_matmultex_matrixW", matrixW);
    llamaShader.SetBuffer(kernel, "matmultex_vectorX", vectorX);
    llamaShader.SetBuffer(kernel, "matmultex_vectorOut", vectorOut);
    llamaShader.SetInt("matmultex_rows", rows);
    llamaShader.SetInt("matmultex_cols", cols);

    int threadGroupsX = Mathf.CeilToInt(rows / 256.0f);
    llamaShader.Dispatch(kernel, threadGroupsX, 1, 1);
    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[vectorOut.ElementCount<float>()];
      vectorOut.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void Accumulate(ComputeBuffer bufferA, ComputeBuffer bufferB, int length) {
    Profiler.BeginSample("accumulate");
    int vecLen = ComputeUtils.GetVectorizedLength(length);

    llamaShader.SetBuffer(_kernels.accumulate, "accumulate_A", bufferA);
    llamaShader.SetBuffer(_kernels.accumulate, "accumulate_B", bufferB);
    llamaShader.SetInt("accumulate_veclen", vecLen);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    llamaShader.Dispatch(_kernels.accumulate, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[bufferA.ElementCount<float>()];
      bufferA.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void RmsNorm(ComputeBuffer bufferIn, ComputeBuffer bufferWeight, ComputeBuffer resultBuffer, int length) {
    Profiler.BeginSample("RmsNorm");

    int vecLen = ComputeUtils.GetVectorizedLength(length);

    llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_In", bufferIn);
    llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_Weight", bufferWeight);
    llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_Out", resultBuffer);
    llamaShader.SetInt("rmsnorm_veclen", vecLen);
    llamaShader.SetFloat("rmsnorm_length", length);

    llamaShader.Dispatch(_kernels.rmsNorm, 1, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[resultBuffer.ElementCount<float>()];
      resultBuffer.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void Rope(ComputeBuffer q, ComputeBuffer k, int pos, int l) {
    Profiler.BeginSample("rope");

    int headSize = _config.dim / _config.n_heads;
    int vecLen = _config.dim / 2;

    llamaShader.SetBuffer(_kernels.rope, "rope_q", q);
    llamaShader.SetBuffer(_kernels.rope, "rope_k", k);
    llamaShader.SetBuffer(_kernels.rope, "rope_freq_cis", _weightsGPU.freq_cis);
    llamaShader.SetInt("rope_freq_cis_offset", pos * headSize / 2);
    llamaShader.SetInt("rope_stride", headSize / 2);
    llamaShader.SetInt("rope_length", vecLen);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    llamaShader.Dispatch(_kernels.rope, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      half[] freq_cis = new half[_weightsGPU.freq_cis.count * 4];
      _weightsGPU.freq_cis.GetData(freq_cis);

      float[] qData = new float[q.ElementCount<float>()];
      q.GetData(qData);
      string debugString = string.Join(", ", new ArraySegment<float>(qData, 0, 8));
      Debug.Log("Rope Q: " + debugString);

      float[] kData = new float[k.ElementCount<float>()];
      k.GetData(kData);
      debugString = string.Join(", ", new ArraySegment<float>(kData, 0, 8));
      Debug.Log("Rope K: " + debugString);
    }
  }

  private void ComputeAttention(ComputeBuffer q, ComputeBuffer k, ComputeBuffer att, int head, int pos) {
    Profiler.BeginSample("computeAttention");

    // Set the buffers
    llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_q", q);
    llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_k", k);
    llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_att", att);

    int headSize = _config.dim / _config.n_heads;

    // Set the variables
    llamaShader.SetInt("compute_attention_head", head);
    llamaShader.SetInt("compute_attention_head_size", headSize);
    llamaShader.SetInt("compute_attention_pos", pos);
    llamaShader.SetInt("compute_attention_dim", _config.dim);
    llamaShader.SetInt("compute_attention_seq_len", _config.seq_len);
    llamaShader.SetFloat("compute_attention_head_size_inv_sqrt", 1.0f / Mathf.Sqrt(headSize));

    // Dispatch the kernel
    // note: we dispatch the entire sequence length because we want to write 0 for the extra numbers
    // todo: get rid of this!  we shouldn't be using them anyway!
    int threadGroupsX = Mathf.CeilToInt(_config.seq_len / 256.0f);
    llamaShader.Dispatch(_kernels.computeAttention, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] qData = new float[q.ElementCount<float>()];
      q.GetData(qData);
      float[] qRow = new float[headSize];
      Array.Copy(qData, head * headSize, qRow, 0, headSize);

      float[] kData = new float[k.ElementCount<float>()];
      k.GetData(kData);
      float[][] kRows = new float[pos + 1][];
      for (int p = 0; p <= pos; ++p) {
        kRows[p] = new float[headSize];
        Array.Copy(kData, head * headSize + p * _config.dim, kRows[p], 0, headSize);
      }

      float[] cpuAtt = new float[pos + 1];
      for (int t = 0; t <= pos; ++t) {
        float score = 0;
        for (int i = 0; i < headSize; ++i) {
          score += qRow[i] * kRows[t][i];
        }

        score *= 1.0f / Mathf.Sqrt(headSize);
        cpuAtt[t] = score;
      }

      float[] resultData = new float[att.ElementCount<float>()];
      att.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, head * _config.seq_len, 8));
      Debug.Log(debugString);
    }
  }
  
  private void Softmax(ComputeBuffer bufferInOut, int offset, int length) {
    Profiler.BeginSample("softmax");

    ComputeBuffer maxBuffer = _runState.scalarTemp0;
    ComputeBuffer sumBuffer = _runState.scalarTemp1;
    
    // First find the max value (as fixed point)
    Profiler.BeginSample("softmax_findmax");
    FindMaxValue(bufferInOut, maxBuffer, offset, length);
    Profiler.EndSample();

    // Next compute exponent and sum of exponents
    Profiler.BeginSample("softmax_exp");
    llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_input", bufferInOut);
    llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_output", _runState.softmaxTemp);
    llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_max_fixed", maxBuffer);
    llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_sum_fixed", sumBuffer);
    llamaShader.SetInt("softmax_offset", offset);
    llamaShader.SetInt("softmax_length", length);
    
    sumBuffer.SetData(new int[] { 0 });

    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    llamaShader.Dispatch(_kernels.softmaxExp, threadGroupsX, 1, 1);
    Profiler.EndSample();
    
    // Finally, divide by sum to get softmax
    Profiler.BeginSample("softmax_divide");
    llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_input", _runState.softmaxTemp);
    llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_output", bufferInOut);
    llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_sum_fixed", sumBuffer);
    llamaShader.SetInt("softmax_offset", offset);
    llamaShader.SetInt("softmax_length", length);
    
    llamaShader.Dispatch(_kernels.softmaxDivide, threadGroupsX, 1, 1);
    {Profiler.EndSample();}

    Profiler.EndSample();

    if (_Debug) {
      float[] tempData = new float[_runState.softmaxTemp.ElementCount<float>()];
      _runState.softmaxTemp.GetData(tempData);

      float[] resultData = new float[bufferInOut.ElementCount<float>()];
      bufferInOut.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, offset, 8));
      Debug.Log(debugString);
    }
  }

  private void Silu(ComputeBuffer bufferInOut, int length) {
    Profiler.BeginSample("silu");
    llamaShader.SetBuffer(_kernels.silu, "silu_InOut", bufferInOut);
    llamaShader.SetInt("silu_length", length);

    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    llamaShader.Dispatch(_kernels.silu, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[bufferInOut.ElementCount<float>()];
      bufferInOut.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void Multiply(ComputeBuffer bufferA, ComputeBuffer bufferB, ComputeBuffer resultBuffer, int length) {
    Profiler.BeginSample("multiply");

    llamaShader.SetBuffer(_kernels.multiply, "multiply_A", bufferA);
    llamaShader.SetBuffer(_kernels.multiply, "multiply_B", bufferB);
    llamaShader.SetInt("multiply_length", length);

    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    llamaShader.Dispatch(_kernels.multiply, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[resultBuffer.ElementCount<float>()];
      resultBuffer.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void WeightedSum(ComputeBuffer valuesBuffer, ComputeBuffer attentionBuffer, ComputeBuffer resultBuffer,
    int head, int pos) {
    Profiler.BeginSample("weightedSum");

    int headSize = _config.dim / _config.n_heads;
    int offset = head * headSize;
    int attentionOffset = head * _config.seq_len;

    llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_values", valuesBuffer);
    llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_attention", attentionBuffer);
    llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_out", resultBuffer);
    llamaShader.SetInt("weightedsum_offset", offset);
    llamaShader.SetInt("weightedsum_attention_offset", attentionOffset);
    llamaShader.SetInt("weightedsum_head_size", headSize);
    llamaShader.SetInt("weightedsum_pos", pos);
    llamaShader.SetInt("weightedsum_dim", _config.dim);

    int threadGroupsX = Mathf.CeilToInt((pos + 1) / 256.0f);
    llamaShader.Dispatch(_kernels.weightedSum, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] resultData = new int[resultBuffer.ElementCount<float>()];
      resultBuffer.GetData(resultData);
      float[] floatData = new float[resultBuffer.ElementCount<float>()];
      for (int i = 0; i < resultBuffer.ElementCount<float>(); ++i) {
        floatData[i] = resultData[i] / (256.0f * 256.0f * 256.0f);
      }

      string debugString = string.Join(", ", new ArraySegment<float>(floatData, offset, 8));
      Debug.Log(debugString);
    }
  }

  private void FindMaxIndex(ComputeBuffer sourceBuffer, ComputeBuffer resultBuffer, int inputLength) {
    float[] checkInput = new float[inputLength];
    sourceBuffer.GetData(checkInput);
    
    llamaShader.SetBuffer(_kernels.findMaxIdx, "findmaxidx_values", sourceBuffer);
    llamaShader.SetBuffer(_kernels.findMaxIdx, "findmaxidx_output", resultBuffer);
    llamaShader.SetInt("findmaxidx_length", inputLength);
    
    int threadGroupsX = Mathf.CeilToInt(inputLength / 256.0f);
    llamaShader.Dispatch(_kernels.findMaxIdx, threadGroupsX, 1, 1);

    int[] resultData = new int[1];
    resultBuffer.GetData(resultData);
  }

  private void FindMaxValue(ComputeBuffer sourceBuffer, ComputeBuffer resultBuffer, int offset, int inputLength) {
    resultBuffer.SetData(new float[] { -100 * 256 * 256 * 256 });
    llamaShader.SetBuffer(_kernels.findMaxVal, "findmaxval_input", sourceBuffer);
    llamaShader.SetBuffer(_kernels.findMaxVal, "findmaxval_output", resultBuffer);
    llamaShader.SetInt("findmaxval_offset", offset);
    llamaShader.SetInt("findmaxval_length", inputLength);
  
    int threadGroupsX = Mathf.CeilToInt(inputLength / 256.0f);
    llamaShader.Dispatch(_kernels.findMaxVal, threadGroupsX, 1, 1);
  }

  private void SampleLogits(ComputeBuffer runStateLogits, float random) {
    Profiler.BeginSample("SampleLogits");

    llamaShader.SetBuffer(_kernels.sampleLogits, "sample_probabilities", runStateLogits);
    llamaShader.SetBuffer(_kernels.sampleLogits, "sample_result", _runState.outputToken);
    llamaShader.SetInt("sample_length", _config.vocab_size);
    llamaShader.SetFloat("sample_random", random);

    llamaShader.Dispatch(_kernels.sampleLogits, 1, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] resultData = new int[1];
      _runState.outputToken.GetData(resultData);
      int resultToken = resultData[0];
      Debug.Log($"Resulting token: {resultToken}");
    }
  }

  public bool LoadWeights(string weightsPath) {
    if (!File.Exists(weightsPath)) {
      weightsPath = Path.Combine(Application.streamingAssetsPath, "models", weightsPath);
    }
    
    float startTime = Time.realtimeSinceStartup;
    Debug.Log("Loading weights...");
    try {
      Profiler.BeginSample("LoadWeights");

      using (FileStream fs = new FileStream(weightsPath, FileMode.Open, FileAccess.Read))
      using (BinaryReader br = new BinaryReader(fs)) {
        // Read the config
        _config = new LlamaConfig(QuantizationMode) {
          dim = br.ReadInt32(),
          hidden_dim = br.ReadInt32(),
          n_layers = br.ReadInt32(),
          n_heads = br.ReadInt32(),
          n_kv_heads = br.ReadInt32(),
          vocab_size = br.ReadInt32(),
          seq_len = br.ReadInt32()
        };

        _config.vocab_size = Math.Abs(_config.vocab_size);

        // Initialize weights
        _weights = new Weights(_config);

        // Read token_embedding_table
        ReadNativeArray(br, _weights.token_embedding_table);

        // Read rms_att_weight for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].rms_att_weight);
        }

        // Read wq for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].wq);
        }

        // Read wk for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].wk);
        }

        // Read wv for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].wv);
        }

        // Read wo for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].wo);
        }

        // Read rms_ffn_weight for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].rms_ffn_weight);
        }

        // Read w1 for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].w1);
        }

        // Read w2 for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].w2);
        }

        // Read w3 for all layers
        for (int layer = 0; layer < _config.n_layers; layer++) {
          ReadNativeArray(br, _weights.layerWeights[layer].w3);
        }

        // Read remaining weights
        ReadNativeArray(br, _weights.rms_final_weight);
        ReadNativeArray(br, _weights.freq_cis_real);
        ReadNativeArray(br, _weights.freq_cis_imag);

        // Read wcls
        if (!_config.UseSharedVocab) {
          ReadNativeArray(br, _weights.wcls);
        }
      }

      Debug.Log("Weights loaded successfully in " + (Time.realtimeSinceStartup - startTime) + "s");

      return true; // Successfully loaded
    }
    catch (Exception e) {
      Debug.LogError($"Failed to load weights from {weightsPath}: {e}");
      return false; // Failed to load
    }
    finally {
      Profiler.EndSample();
    }
  }

  private void ReadNativeArray(BinaryReader br, NativeArray<float> array) {
    int byteCount = array.Length * sizeof(float);
    byte[] buffer = br.ReadBytes(byteCount);

    // Get a pointer to the NativeArray's data
    unsafe {
      fixed (byte* pBuffer = &buffer[0]) {
        UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafePtr(array), pBuffer, byteCount);
      }
    }
  }

  private void PrintLogitsDebug() {
    float[] logits = new float[_config.vocab_size];
    _runState.logits.GetData(logits);

    List<Tuple<int, float>> sortedLogits = new List<Tuple<int, float>>();
    for (int i = 0; i < _config.vocab_size; i++) {
      sortedLogits.Add(new Tuple<int, float>(i, logits[i]));
    }

    sortedLogits.Sort((a, b) => b.Item2.CompareTo(a.Item2));

    for (int i = 0; i < 10; i++) {
      Tuple<int, float> token = sortedLogits[i];
      string tokenString = Tokenizer.Detokenize(token.Item1);
      Debug.Log($"Top {i}: {tokenString} {token.Item2}");
    }

    int[] outputToken = new int[1];
    _runState.outputToken.GetData(outputToken);
    string outputTokenString = Tokenizer.Detokenize(outputToken[0]);

    string debugString = string.Join(", ", new ArraySegment<float>(logits, 0, 256));
    Debug.Log($"Got output token {outputTokenString} with logits {debugString}");
  }

  public static RenderTexture CreateWeightsTexture(QuantizationModes mode, int rows, int columns) {
    var format = QuantizationUtil.QuantizationFormats[mode];
    RenderTextureFormat WeightsFormat = format;
    return new RenderTexture(columns, rows, 0, WeightsFormat, RenderTextureReadWrite.Linear) {
      enableRandomWrite = true,
      filterMode = FilterMode.Point,
      wrapMode = TextureWrapMode.Clamp,
      useMipMap = false,
    };
  }
}