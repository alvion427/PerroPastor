# PerroPastor
With all of these latin american wild animals running around (Llamas, Alpacas, Guanacos, Vicuñas) we need a good Perro Pastor ("sheep dog") to get them running!  Perro Pastor is a Unity package written with just a few files of C# and compute shaders to run Llama based models on any Unity compatible platform on the gpu!  I built this package primarily as an exercise to understand the execution of LLM inference at a deeper level, but I think that it has the potential to be useful to game developers who want to run low latency small LLMs in their games without relying on server infrastructure.

I also intend this repo to be a good opportunity to learn how LLMs work.  Although compute shaders aren't the most readable form of source code, I have included lots of detailed comments in the C# code which should help demystify what is going on.

# Getting Started
* Clone the repo.
* Make a new Unity project
* Package Manager -> Add Package From Disk -> Choose "package.json" from repo folder.
* A "Perro" menu should appear in Unity, choose "Download Sample Model" to download the model and tokenizer files to StreamingAssets/Models
* Find PerroPastor in your Packages list in the unity Project window, copy the sample scene to your project.
* Run the scene!
* Try entering different prompts in the Llama.Prompt field (they shold be the first few words of a story)
* Check out the [RunTransformer](https://github.com/alvion427/PerroPastor/blob/master/UnityPackage/Llama.cs#L168) function for lots of comments explaining in detail what is going on.
* To run other models, you should be able to run any ggml compatible models in fp32, fp16, or q8_0, or q5_1.  Q5_1 is definitely the preference as far as performance goes (I get 10+ tokens per second on my xps15 with RTX 3050 and 4 gigs of vram).

# Thank you Andrej Karpathy!!!
This project was inspired by Andrej Karpathy's excellent [llama2.c project](https://github.com/karpathy/llama2.c), and owes a huge dept to Andrej and all of his work he has done over the years to support LLM literacy.  I find it so much easier to learn how something REALLY works when it is written in a simple language with no dependencies, rather than trying to peer through all the abstractions of higher level libraries like PyTorch.  You will see that lots of the naming and overall code flow still owes a great deal to llama2.c, although things have already started to diverge quite a bit, and will diverge a lot more in the next few weeks as I make things a bit more conducive to real-time game execution.

If you are new to LLMs, I HIGHLY recommend his [LLM Zero to Hero](https://youtu.be/VMj-3S1tku0) series on youtube, and especially the [Building ChatGPT from Scratch](https://youtu.be/kCc8FmEb1nY) video.  If you already know c and aren't very familiar with compute shaders, you may find it easier to read through his llama2.c project first.

# TODO
* Support a few more ggml quantization formats (q4_1 and q5_m/s are the highest priority).
* LOTS of easy optimizations in the existing kernels.
* Add topk filtering.
* Fuse compute kernels to reduce draw calls and intermediate memory usage.
* Add support for Sentis when they support 8 bit quantization.
* Implement asyncrhonous model loading to reduce memory usage during startup.
* Add support for pipelined execution of multiple instances of the same model.
* Add support for LoRA weights at inference time to enable multiple variations on one base model at runtime.
