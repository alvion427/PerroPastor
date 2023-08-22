# PerroPastor
With all of these latin american wild animals running around (Llamas, Alpacas, Guanacos, Vicunyas) we need a good Perro Pastor ("sheep dog") to get them running!  Perro Pastor is a Unity package written with just a few files of C# and compute shaders to run Llama based models on any Unity compatible platform on the gpu!  I built this package primarily as an excercise to understand the execution of LLM inference at a deeper level, but I think that it has the potential to be useful to game developers who want to run low latency small LLMs in their games without relying on server infrastructure.

I also intend this repo to be a good opportunity to learn how LLMs work.  Although compute shaders aren't the most readable form of source code, I have included lots of detailed comments in the C# code which should help demystify what is going on.

# Status
*Lots* of optimization needed.  Even small LLMs (15m and 110m stories libraries provided as examples) take 50+ ms to run on my xps15 with RTX 3050.  There are a lot of very easy optimizations to make to get things running more smoothly, although 8 bit weight quantization and 16bit inference are probably necessary before you can really do anything useful.

# Features
* Run LLMs in 32 bit or with 16 bit quantization (for weights only).
* Deploy to any Unity platform that supports compute shaders.

# Getting Started
* Clone the repo.
* Make a new Unity project
* Package Manager -> Add Package From Disk -> Choose "package.json" from repo folder.
* A "Perro" menu should appear in Unity, choose "Download Sample Model" to download the model and tokenizer files to StreamingAssets/Models
* Find PerroPastor in your Packages list in the unity Project window, copy the sample scene to your project.
* Run the scene!
* Check out the [RunTransformer](https://github.com/alvion427/PerroPastor/blob/master/UnityPackage/Llama.cs#L168) function for lots of comments explaining in detail what is going on.

# Thank you Andrej Karpathy!!!
This project was inspired by Andrej Karpathy's excellent [llama.c project](https://github.com/karpathy/llama2.c), and owes a huge dept to Andrej and all of his work he has done over the years to support LLM literacy.  I find it so much easier to learn how something REALLY works when it is written in a simple language with no dependencies, rather than trying to peer through all the abstractions of higher level libraries like PyTorch.  You will see that lots of the naming and overall code flow still owes a great deal to llama.c, although things have already started to diverge quite a bit, and will diverge a lot more in the next few weeks as I make things a bit more conducive to real-time game execution.

If you are new to LLMs, I HIGHLY recommend his [LLM Zero to Hero](https://youtu.be/VMj-3S1tku0) series on youtube, and especially the [Building ChatGPT from Scratch](https://youtu.be/kCc8FmEb1nY) video.  If you already know c and aren't very familiar with compute shaders, you may find it easier to 

# TODO
* Parallelize the compute kernels that are currently running in serial (RmsNorm, Softmax and SampleLogits).
* Add support for 16 bit inference (especially for the kv cache).
* Fuse compute kernels to reduce draw calls and intermediate memory usage.
* Implement asyncrhonous model loading to reduce memory usage during startup.
* Implement LLM.int8() for 8 bit inference.
* *(at this point we should be able to run Llama13b on most desktops and laptops)*
* Add support for pipelined execution of multiple instances of the same model.
* Add support for LoRA weights at inference time to enable multiple variations on one base model at runtime.
