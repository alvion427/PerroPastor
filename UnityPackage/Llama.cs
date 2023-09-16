using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

public class Llama : MonoBehaviour {
  public int RandomSeed = -1; // -1 means use time

  public Tokenizer Tokenizer => _tokenizer;
  public LlamaConfig Config => _config;
  public QuantizationModes RuntimeQuantizationMode = QuantizationModes.Float32;
  public ModelLoaderBase ModelLoader;
  private ComputeShader _llamaShader;

  public int MaxTokensPerFrame = 1;

  private List<Conversation> _conversations = new List<Conversation>();

  public string SaveTrace;
  public string CheckTrace;

  private bool _isInitialized = false;
  private LlamaConfig _config;
  private WeightsGpu _weightsGpu;
  private RunState _runState;
  private System.Random _rng;
  private GpuStateDebugger _gpuStateDebugger;
  private LlamaKernels _kernels;
  private Tokenizer _tokenizer; 

  public bool _Debug = false;

  void Start() {
    // We need to reset any static state to allow us to run in editor without domain reloading.
    ComputeUtils.Reset();
    TextureUtils.Reset();
    
    if (!string.IsNullOrEmpty(CheckTrace)) {
      _gpuStateDebugger = new GpuStateDebugger(GpuStateDebugger.Modes.Check, CheckTrace);
    }
    else if (!string.IsNullOrEmpty(SaveTrace)) {
      _gpuStateDebugger = new GpuStateDebugger(GpuStateDebugger.Modes.Save, SaveTrace);
    }

    _rng = new System.Random(RandomSeed == -1 ? (int)DateTime.Now.Ticks : RandomSeed);

    LoadShader();

    if (ModelLoader == null) {
      ModelLoader = GetComponents<ModelLoaderBase>().FirstOrDefault(comp => comp.enabled);
    } 
    ModelLoader.OnLoaded += (config, weightsGpu, tokenizer) => {
      _config = config;
      _weightsGpu = weightsGpu;
      _tokenizer = tokenizer;
      
      Initialize();
    };
      
    ModelLoader.RequestLoad();
  }

  private void OnDestroy() {
    if (_isInitialized) {
      Uninitialize();
    }
  }

  public void StartConversation(Conversation conversation) {
    _conversations.Add(conversation);
  }

  void Update() {
    if (!_isInitialized) {
      return;
    }
    
    foreach (Conversation c in _conversations)
    {
      int tokensProcessedThisFrame = 0;
      while (!c._sequenceComplete && c._tokensToRun > 0 && tokensProcessedThisFrame < MaxTokensPerFrame) {
        ++tokensProcessedThisFrame;

        RunTransformer(c);
      
        bool isFinalToken = c._tokensToRun == 1;

        if (c._pos < c._resultTokens.Count) {
          // This is still part of the prompt, so we already know the token result
          int queryToken = c._resultTokens[c._pos];
          c._outputToken.SetData(new int[] { queryToken });
          c.ProduceToken(c._pos, queryToken, isFinalToken);
        }
        else {
          if (c.Temperature == 0) {
            FindMaxIndex(_runState.logits, c._outputToken, _config.vocab_size);
          }
          else {
            ScaleBuffer(_runState.logits, 1.0f / c.Temperature, _config.vocab_size);
            Softmax(_runState.logits, 0, _config.vocab_size);
            SampleLogits(c._outputToken, (float)_rng.NextDouble());
          }

          int pos = c._pos;
          AsyncGPUReadback.Request(c._outputToken, (request) => {
            if (c._sequenceComplete) {
              return;
            }
          
            if (request.hasError) {
              Debug.LogError("Failed to readback output token buffer");
              enabled = false;
              return;
            }

            Debug.Assert(c._resultTokens.Count < _config.seq_len);

            int token = request.GetData<int>()[0];
            if (token == _tokenizer.SOS || token == _tokenizer.EOS) {
              Debug.Log("Sequence ended with sos/eos");
              SequenceComplete(c, pos - 1);
              return;
            }

            c.ProduceToken(pos, token, isFinalToken);
          });
        }

        --c._tokensToRun;
        ++c._pos;
      }
      
    }
  }

  internal void SequenceComplete(Conversation conversation, int finalPos) {
    _gpuStateDebugger?.TraceFinished();
    conversation.SequenceComplete(finalPos);
  }

  private void LoadShader() {
    if (_llamaShader == null)
      _llamaShader = (ComputeShader)Resources.Load("Llama");
    _kernels = new LlamaKernels(_llamaShader);
  }

  public void RunTransformer(Conversation conversation) {
    if (!_isInitialized) {
      Initialize();
    }

    int pos = conversation._pos;
    int dim = _config.dim;
    int hidden_dim = _config.hidden_dim;
    
    // The first step is to load the embedding for the current token.  This is just a simple lookup into the
    // embedding table of the last generated token (or the start of sequence token when we begin).
    LoadEmbedding(_runState.x, conversation._outputToken, conversation._pos);
    _gpuStateDebugger?.ProcessState($"token{pos}_load_embedding", _runState.x);

    // We process each layer more or less independently, and the results of each layer feed into the next layer.
    // The input for each layer is in runState.x, which initially is just the embedding of the input token.
    // At the end of each layer, we write the layers output (along with a residual connection to the input) back
    // into x to serve as the input for the next layer.
    for (int l = 0; l < _config.n_layers; l++) {
      // Unlike the original Transformer architecture, which scaled the output of attention by a LayerNorm, Llama
      // pre-scales the inputs to the layers using RMS (root-mean-squared) normalization.  
      // NOTE!  RMSNorm also sneaks in a scaling by a set of weights.  These weights are defined per-layer, and
      // allow each layer to emphasize different features in the input.
      RmsNorm(_runState.x, _weightsGpu.layerWeights[l].rms_att_weight, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_rmsnorm", _runState.xb);

      // QKV (query, key, value) matmuls for this position.
      // The normalized layer input is multiplied by (dim, dim) matrices wq, wk, and wv, the results of which
      // are stored in the (dim,) vectors q, k, and v respectively.  You can think of this at the query, key, and
      // value vectors for the current token.
      MatMul(_weightsGpu.layerWeights[l].wq, _runState.xb, _runState.q, dim, dim);
      MatMul(_weightsGpu.layerWeights[l].wk, _runState.xb, _runState.k, dim, dim);
      MatMul(_weightsGpu.layerWeights[l].wv, _runState.xb, _runState.v, dim, dim);

      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_q", _runState.q);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_k", _runState.k);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_v", _runState.v);

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
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_rope_q", _runState.q);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_rope_k", _runState.k);

      // As we process each token, we accumulate the key and value vectors into a key/value cache, which allows us
      // to reuse them for future tokens.
      // XXX: Copy only bytes using runtime size!
      Memcpy(conversation._persistentState.layers[l].key_cache, _runState.k, 0, pos * dim, dim);
      Memcpy(conversation._persistentState.layers[l].value_cache, _runState.v, 0, pos * dim, dim);

      // The results of attention are an accumulation in this buffer, so we zero it out for all heads before
      // computing attention 
      ClearBuffer(_runState.xbFixed);

      // Multi-head attention. Iterate over each head and compute it's attention
      for (int h = 0; h < _config.n_heads; h++) {
        // First we compute the raw attention scores for this head, stored in vector runState.att (seq_len,).
        // To do this, we simply take the query vector for the current token (stored in runState.q) and then for
        // each previous token in the sequence we dot product runState.q with keyCache.v[pos] and store it in
        // runState.att[pos].
        // XXX: Load keys using 16 bit!
        ComputeAttention(_runState.q, conversation._persistentState.layers[l].key_cache, _runState.att, h, pos);

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
        // XXX: Use 16 bit value cache!!
        WeightedSum(conversation._persistentState.layers[l].value_cache, _runState.att, _runState.xbFixed, h, pos);
      }
      
      // Convert all fixed point output vectors from attention into float
      FixedToFloat(_runState.xbFixed, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_attention", _runState.xb);

      // Final matmul to get the output of the attention.  This allows us to apply some additional learned weights
      // to the output of the attention.
      MatMul(_weightsGpu.layerWeights[l].wo, _runState.xb, _runState.xb2, dim, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_weighted_attention", _runState.xb2);

      // Residual connection back into x.  This allows us to combine many layers without losing signal during
      // backpropagation through many layers.
      Accumulate(_runState.x, _runState.xb2, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_accumulate_attention", _runState.x);

      // A final RMS norm of the outputs of the attention module.
      RmsNorm(_runState.x, _weightsGpu.layerWeights[l].rms_ffn_weight, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_attention_norm", _runState.xb);

      // For the feedforward network, we use a 2-layer MLP with an additional elementwise multiply to allow the
      // network to learn to amplify or diminish the outputs from the first layer (ie gating).  Also unlike
      // the original Transformer, we use SiLU (swish) as the non-linearity.
      // The FFN would be expressed in pytorch as: w2(F.silu(w1(x)) * w3(x))

      // Multiply w1(x) to expand (dim,) to (hidden_dim,)
      MatMul(_weightsGpu.layerWeights[l].w1, _runState.xb, _runState.hb, hidden_dim, dim);

      // Silu nonlinearity
      Silu(_runState.hb, hidden_dim);

      // Multiply w3(x) to calculate gating parameters (hidden_dim,)
      MatMul(_weightsGpu.layerWeights[l].w3, _runState.xb, _runState.hb2, hidden_dim, dim);

      // Elementwise multiply with w3(x) to apply gating (amplify or diminish results of w1(x))
      Multiply(_runState.hb, _runState.hb2, _runState.hb, hidden_dim);

      // Multiply result by w2 to compress back to (dim,)
      MatMul(_weightsGpu.layerWeights[l].w2, _runState.hb, _runState.xb, dim, hidden_dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_ffn", _runState.xb);

      // Again, apply a residual connection back to the original layer input to preserve the gradient.
      Accumulate(_runState.x, _runState.xb, dim);
      _gpuStateDebugger?.ProcessState($"token{pos}_layer{l}_accumulate_ffn", _runState.x);
    }
    
    // Final rmsnorm of all layer results
    RmsNorm(_runState.x, _weightsGpu.rms_final_weight, _runState.xb, dim);
    _gpuStateDebugger?.ProcessState($"token{pos}_final_rmsnorm", _runState.xb);

    // Classify normalized results into logits with one big, fat matmul
    MatMul(_weightsGpu.GetWCLS(), _runState.xb, _runState.logits, _config.vocab_size, dim);
    _gpuStateDebugger?.ProcessState($"token{pos}_classifier", _runState.logits);
  }

  private void Initialize() {
    _isInitialized = true;

    QuantizationUtil.EnableQuantizationKeywords(_llamaShader, RuntimeQuantizationMode, "RUNTIME");
    
    _runState = new RunState(_config, RuntimeQuantizationMode);
    
    foreach (Conversation c in _conversations) {
      c.Initialize();
    }
  }

  private void Uninitialize() {
    // TODO: Wait for finishing all tasks?

    _runState.Dispose();

    _weightsGpu.RemoveReference();

    _isInitialized = false;
  }

  private void ClearBuffer(ComputeBuffer dest) {
    Profiler.BeginSample("ClearBuffer");

    int length = dest.count;
    _llamaShader.SetBuffer(_kernels.clear, "clear_dest", dest);
    _llamaShader.SetInt("clear_length", length);
    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    _llamaShader.Dispatch(_kernels.clear, threadGroupsX, 1, 1);

    Profiler.EndSample();
  }

  private void Memcpy(ComputeBuffer copy_dest, ComputeBuffer copy_source, int sourceOffset, int destOffset,
    int length) {
    Profiler.BeginSample("Memcpy");

    int vecLen = ComputeUtils.GetVectorizedLength(length);
    sourceOffset = ComputeUtils.GetVectorizedLength(sourceOffset);
    destOffset = ComputeUtils.GetVectorizedLength(destOffset);

    // Set the buffers
    _llamaShader.SetBuffer(_kernels.memcpy, "copy_dest", copy_dest);
    _llamaShader.SetBuffer(_kernels.memcpy, "copy_source", copy_source);

    // Set the length
    _llamaShader.SetInt("memcpy_source_offset", sourceOffset);
    _llamaShader.SetInt("memcpy_dest_offset", destOffset);
    _llamaShader.SetInt("memcpy_veclen", vecLen);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _llamaShader.Dispatch(_kernels.memcpy, threadGroupsX, 1, 1);

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

    _llamaShader.SetBuffer(_kernels.scaleBuffer, "scalebuffer_buffer", buffer);
    _llamaShader.SetInt("scalebuffer_veclen", vecLen);
    _llamaShader.SetFloat("scalebuffer_scale", scale);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _llamaShader.Dispatch(_kernels.scaleBuffer, threadGroupsX, 1, 1);

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
    _llamaShader.SetBuffer(_kernels.fixedToFloat, "fixedtofloat_source", fixedBuffer);
    _llamaShader.SetBuffer(_kernels.fixedToFloat, "fixedtofloat_dest", floatBuffer);

    // Set the length
    _llamaShader.SetInt("fixedtofloat_length", vecLength);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(vecLength / 256.0f);
    _llamaShader.Dispatch(_kernels.fixedToFloat, threadGroupsX, 1, 1);

    Profiler.EndSample();
  }

  private void LoadEmbedding(ComputeBuffer embedding, ComputeBuffer token, int pos) {
    Profiler.BeginSample("loadEmbedding");

    int blockCount = _config.dim / _weightsGpu.token_embedding_table.BlockSize;

    // Set the buffers
    _llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_token", token);
    _llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_source", _weightsGpu.token_embedding_table);
    _llamaShader.SetBuffer(_kernels.loadEmbedding, "loadembedding_dest", embedding);
    _llamaShader.SetInt("loadembedding_blockCount", blockCount);

    // Dispatch the kernel
    int threadGroupsX = Mathf.CeilToInt(blockCount / 256.0f);
    _llamaShader.Dispatch(_kernels.loadEmbedding, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] tokenData = new int[token.ElementCount<int>()];
      token.GetData(tokenData);

      float[] resultData = new float[embedding.ElementCount<float>()];
      embedding.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void MatMul(GpuTensor matrixW, ComputeBuffer vectorX, ComputeBuffer vectorOut, int rows, int cols) {
    Profiler.BeginSample("Matmul");

    int blocksPerRow = cols / matrixW.BlockSize;

    // W (d,n) @ x (n,) -> xout (d,)
    _llamaShader.SetBuffer(_kernels.matmul, "matmul_matrixW", matrixW);
    _llamaShader.SetBuffer(_kernels.matmul, "matmul_vectorX", vectorX);
    _llamaShader.SetBuffer(_kernels.matmul, "matmul_vectorOut", vectorOut);
    _llamaShader.SetInt("matmul_rows", rows);
    _llamaShader.SetInt("matmul_cols", cols);
    _llamaShader.SetInt("matmul_blocksPerRow", blocksPerRow);
    
    int threadGroupsX = Mathf.CeilToInt(rows / 128.0f);
    _llamaShader.Dispatch(_kernels.matmul, threadGroupsX, 1, 1);
    Profiler.EndSample();

    if (_Debug) {
      float[] inputData = new float[vectorX.ElementCount<float>()];
      vectorX.GetData(inputData);

      float[] resultData = new float[vectorOut.ElementCount<float>()];
      vectorOut.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void Accumulate(ComputeBuffer bufferA, ComputeBuffer bufferB, int length) {
    Profiler.BeginSample("accumulate");
    int vecLen = ComputeUtils.GetVectorizedLength(length);

    _llamaShader.SetBuffer(_kernels.accumulate, "accumulate_A", bufferA);
    _llamaShader.SetBuffer(_kernels.accumulate, "accumulate_B", bufferB);
    _llamaShader.SetInt("accumulate_veclen", vecLen);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _llamaShader.Dispatch(_kernels.accumulate, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      float[] resultData = new float[bufferA.ElementCount<float>()];
      bufferA.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, 0, 8));
      Debug.Log(debugString);
    }
  }

  private void RmsNorm(ComputeBuffer bufferIn, GpuTensor rmsWeights, ComputeBuffer resultBuffer, int length) {
    Profiler.BeginSample("RmsNorm");

    int vecLen = ComputeUtils.GetVectorizedLength(length);
    int blockCount = _config.dim / _weightsGpu.token_embedding_table.BlockSize;

    _llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_In", bufferIn);
    _llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_Weight", rmsWeights);
    _llamaShader.SetBuffer(_kernels.rmsNorm, "rmsnorm_Out", resultBuffer);
    _llamaShader.SetInt("rmsnorm_vecLen", vecLen);
    _llamaShader.SetInt("rmsnorm_blockCount", blockCount);
    _llamaShader.SetFloat("rmsnorm_length", length);

    _llamaShader.Dispatch(_kernels.rmsNorm, 1, 1, 1);

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

    _llamaShader.SetBuffer(_kernels.rope, "rope_q", q);
    _llamaShader.SetBuffer(_kernels.rope, "rope_k", k);
    _llamaShader.SetInt("rope_head_size", headSize);
    _llamaShader.SetInt("rope_pos", pos);
    _llamaShader.SetInt("rope_length", vecLen);

    int threadGroupsX = Mathf.CeilToInt(vecLen / 256.0f);
    _llamaShader.Dispatch(_kernels.rope, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
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

  private unsafe void ComputeAttention(ComputeBuffer q, ComputeBuffer k, ComputeBuffer att, int head, int pos) {
    Profiler.BeginSample("computeAttention");
    
    // Set the buffers
    _llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_q", q);
    _llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_k", k);
    _llamaShader.SetBuffer(_kernels.computeAttention, "compute_attention_att", att);

    int headSize = _config.dim / _config.n_heads;
    int headSizeVec = ComputeUtils.GetVectorizedLength(headSize);
    int dimVec = ComputeUtils.GetVectorizedLength(_config.dim);

    // Set the variables
    _llamaShader.SetInt("compute_attention_head", head);
    _llamaShader.SetInt("compute_attention_head_size_vec", headSizeVec);
    _llamaShader.SetInt("compute_attention_pos", pos);
    _llamaShader.SetInt("compute_attention_dim_vec", dimVec);
    _llamaShader.SetInt("compute_attention_seq_len", _config.seq_len);
    _llamaShader.SetFloat("compute_attention_head_size_inv_sqrt", 1.0f / Mathf.Sqrt(headSize));

    int threadGroupsX = Mathf.CeilToInt(pos + 1 / 256.0f);
    _llamaShader.Dispatch(_kernels.computeAttention, threadGroupsX, 1, 1);
    
    Profiler.EndSample();

    if (_Debug) {
      float[] qData = new float[q.ElementCount<float>()];
      q.GetData(qData);
      float[] qRow = new float[headSize];
      Array.Copy(qData, head * headSize, qRow, 0, headSize);

      float[] resultData = new float[att.ElementCount<float>()];
      att.GetData(resultData);
      string debugString = string.Join(", ", new ArraySegment<float>(resultData, head * _config.seq_len, 8));
      Debug.Log(debugString);
    }
  }
  
  private void Softmax(ComputeBuffer bufferInOut, int offset, int length) {
    const int kSoftmaxStride = 8;
    
    Profiler.BeginSample("softmax");
    
    ComputeBuffer maxBuffer = _runState.scalarTemp0;
    ComputeBuffer sumBuffer = _runState.scalarTemp1;
    
    // First find the max value (as fixed point)
    Profiler.BeginSample("softmax_findmax");
    FindMaxValue(bufferInOut, maxBuffer, offset, length);
    Profiler.EndSample();
    
    // Next compute exponent and sum of exponents
    Profiler.BeginSample("softmax_exp");
    _llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_input", bufferInOut);
    _llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_output", _runState.softmaxTemp);
    _llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_max_fixed", maxBuffer);
    _llamaShader.SetBuffer(_kernels.softmaxExp, "softmax_sum_fixed", sumBuffer);
    _llamaShader.SetInt("softmax_offset", offset);
    _llamaShader.SetInt("softmax_length", length);

    int numBatches = Mathf.CeilToInt(length / (float)kSoftmaxStride);
    _llamaShader.SetInt("softmax_numBatches", numBatches);
    
    sumBuffer.SetData(new int[] { 0 });

    int threadGroupsExp = Mathf.CeilToInt(numBatches / 256.0f);
    _llamaShader.Dispatch(_kernels.softmaxExp, threadGroupsExp, 1, 1);
    Profiler.EndSample();
    
    // Finally, divide by sum to get softmax
    Profiler.BeginSample("softmax_divide");
    _llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_input", _runState.softmaxTemp);
    _llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_output", bufferInOut);
    _llamaShader.SetBuffer(_kernels.softmaxDivide, "softmax_sum_fixed", sumBuffer);
    _llamaShader.SetInt("softmax_offset", offset);
    _llamaShader.SetInt("softmax_length", length);
    
    int threadGroupsDiv = Mathf.CeilToInt(length / 256.0f);
    _llamaShader.Dispatch(_kernels.softmaxDivide, threadGroupsDiv, 1, 1);
    {Profiler.EndSample();}

    Profiler.EndSample();

    if (_Debug) {
      int[] sumData = new int[1];
      sumBuffer.GetData(sumData);
      float sum = (float)sumData[0] / (256.0f * 256.0f);

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
    _llamaShader.SetBuffer(_kernels.silu, "silu_InOut", bufferInOut);
    _llamaShader.SetInt("silu_length", length);

    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    _llamaShader.Dispatch(_kernels.silu, threadGroupsX, 1, 1);

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

    _llamaShader.SetBuffer(_kernels.multiply, "multiply_A", bufferA);
    _llamaShader.SetBuffer(_kernels.multiply, "multiply_B", bufferB);
    _llamaShader.SetInt("multiply_length", length);

    int threadGroupsX = Mathf.CeilToInt(length / 256.0f);
    _llamaShader.Dispatch(_kernels.multiply, threadGroupsX, 1, 1);

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

    int headSizeVec = ComputeUtils.GetVectorizedLength(_config.dim / _config.n_heads);
    int offsetVec = head * headSizeVec;
    int attentionOffset = head * _config.seq_len;
    int dimVec = ComputeUtils.GetVectorizedLength(_config.dim);

    _llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_values", valuesBuffer);
    _llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_attention", attentionBuffer);
    _llamaShader.SetBuffer(_kernels.weightedSum, "weightedsum_out", resultBuffer);
    _llamaShader.SetInt("weightedsum_offset_vec", offsetVec);
    _llamaShader.SetInt("weightedsum_attention_offset", attentionOffset);
    _llamaShader.SetInt("weightedsum_head_size_vec", headSizeVec);
    _llamaShader.SetInt("weightedsum_pos", pos);
    _llamaShader.SetInt("weightedsum_dim_vec", dimVec);

    int threadGroupsX = Mathf.CeilToInt((pos + 1) / 256.0f);
    _llamaShader.Dispatch(_kernels.weightedSum, threadGroupsX, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] resultData = new int[resultBuffer.ElementCount<float>()];
      resultBuffer.GetData(resultData);
      float[] floatData = new float[resultBuffer.ElementCount<float>()];
      for (int i = 0; i < resultBuffer.ElementCount<float>(); ++i) {
        floatData[i] = resultData[i] / (256.0f * 256.0f * 256.0f);
      }

      string debugString = string.Join(", ", new ArraySegment<float>(floatData, offsetVec * 4, 8));
      Debug.Log(debugString);
    }
  }

  private void FindMaxIndex(ComputeBuffer sourceBuffer, ComputeBuffer resultBuffer, int inputLength) {
    _llamaShader.SetBuffer(_kernels.findMaxIdx, "findmaxidx_values", sourceBuffer);
    _llamaShader.SetBuffer(_kernels.findMaxIdx, "findmaxidx_output", resultBuffer);
    _llamaShader.SetInt("findmaxidx_length", inputLength);
    
    int threadGroupsX = Mathf.CeilToInt(inputLength / 256.0f);
    _llamaShader.Dispatch(_kernels.findMaxIdx, threadGroupsX, 1, 1);
  }

  private void FindMaxValue(ComputeBuffer sourceBuffer, ComputeBuffer resultBuffer, int offset, int inputLength) {
    resultBuffer.SetData(new float[] { 0 });
    _llamaShader.SetBuffer(_kernels.findMaxVal, "findmaxval_input", sourceBuffer);
    _llamaShader.SetBuffer(_kernels.findMaxVal, "findmaxval_output", resultBuffer);
    _llamaShader.SetInt("findmaxval_offset", offset);
    _llamaShader.SetInt("findmaxval_length", inputLength);
  
    int threadGroupsX = Mathf.CeilToInt(inputLength / 256.0f);
    _llamaShader.Dispatch(_kernels.findMaxVal, threadGroupsX, 1, 1);

    if (_Debug) {
      int[] maxData = new int[1];
      resultBuffer.GetData(maxData);
      float max = (float)maxData[0] / (256.0f * 256.0f);
      Debug.Log($"Max: {max} ({maxData[0]})");
    }
  }

  private void SampleLogits(ComputeBuffer outputToken, float random) {
    Profiler.BeginSample("SampleLogits");

    _llamaShader.SetBuffer(_kernels.sampleLogits, "sample_probabilities", _runState.logits);
    _llamaShader.SetBuffer(_kernels.sampleLogits, "sample_result", outputToken);
    _llamaShader.SetInt("sample_length", _config.vocab_size);
    _llamaShader.SetFloat("sample_random", random);

    _llamaShader.Dispatch(_kernels.sampleLogits, 1, 1, 1);

    Profiler.EndSample();

    if (_Debug) {
      int[] resultData = new int[1];
      outputToken.GetData(resultData);
      int resultToken = resultData[0];
      Debug.Log($"Resulting token: {resultToken}");
    }
  }

  private void PrintLogitsDebug(Conversation conversation) {
    float[] logits = new float[_config.vocab_size];
    _runState.logits.GetData(logits);

    List<Tuple<int, float>> sortedLogits = new List<Tuple<int, float>>();
    for (int i = 0; i < _config.vocab_size; i++) {
      sortedLogits.Add(new Tuple<int, float>(i, logits[i]));
    }

    sortedLogits.Sort((a, b) => b.Item2.CompareTo(a.Item2));

    for (int i = 0; i < 10; i++) {
      Tuple<int, float> token = sortedLogits[i];
      string tokenString = _tokenizer.Detokenize(token.Item1);
      Debug.Log($"Top {i}: {tokenString} {token.Item2}");
    }

    int[] outputToken = new int[1];
    conversation._outputToken.GetData(outputToken);
    string outputTokenString = _tokenizer.Detokenize(outputToken[0]);

    string debugString = string.Join(", ", new ArraySegment<float>(logits, 0, 256));
    Debug.Log($"Got output token {outputTokenString} with logits {debugString}");
  }
}