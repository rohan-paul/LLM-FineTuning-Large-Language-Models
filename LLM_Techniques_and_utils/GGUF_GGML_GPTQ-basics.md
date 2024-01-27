```py

# GGML or GGUF in the world of Large Language Models

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load LLM and Tokenizer
model_id = "TheBloke/zephyr-7B-beta-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=False,
    revision="main"
)

# Create a pipeline
pipe = pipeline(model=model, tokenizer=tokenizer, task='text-generation')


###############################################

# AWQ: Activation-aware Weight Quantization

!pip install vllm

from vllm import LLM, SamplingParams

# Load the LLM
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=256)
llm = LLM(
    model="TheBloke/zephyr-7B-beta-AWQ",
    quantization='awq',
    dtype='half',
    gpu_memory_utilization=.95,
    max_model_len=4096
)

```


## 🚀 What is GGML or GGUF in the world of Large Language Models ? 🚀

GGUF / GGML are file formats for quantized models

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML.

Basically, GGUF (i.e. "GPT-Generated Unified Format"), previously GGML, is a quantization method that allows users to use the CPU to run an LLM but also offload some of its layers to the GPU for a speed up.

📌 GGML is a C++ Tensor library designed for machine learning, facilitating the running of LLMs either on a CPU alone or in tandem with a GPU.

💡 GGUF (new)

💡 GGML (old)

Llama.cpp has dropped support for the GGML format and now only supports GGUF

------------

* GGUF contains all the metadata it needs in the model file (no need for other files like tokenizer_config.json) except the prompt template

* llama.cpp has a script to convert `*.safetensors` model files into `*.gguf`

* Transformers & Llama.cpp support both CPU, GPU and MPU inference

Being compiled in C++, with GGUF the inference is multithreaded.

↪️ GGML format recently changed to GGUF which is designed to be extensible, so that new features shouldn’t break compatibility with existing models. It also centralizes all the metadata in one file, such as special tokens, RoPE scaling parameters, etc. In short, it answers a few historical pain points and should be future-proof.

----------------

📌 GGUF (GGML) vs GPTQ

▶️ GPTQ is not the same quantization format as GGUF/GGML. They are different approaches with different codebases but have borrowed ideas from each other.

▶️ GPTQ is a post-training quantziation method to compress LLMs, like GPT. GPTQ compresses GPT models by reducing the number of bits needed to store each weight in the model, from 32 bits down to just 3-4 bits.

▶️ GPTQ analyzes each layer of the model separately and approximating the weights in a way that preserves the overall accuracy.

▶️ Quantizes the weights of the model layer-by-layer to 4 bits instead of 16 bits, this reduces the needed memory by 4x.

▶️ Achieves same latency as fp16 model, but 4x less memory usage, sometimes faster due to custom kernels, e.g. Exllama

----------------------------

▶️ There's also the bits and bytes library, which quantizes on the fly (to 8-bit or 4-bit) and is related to QLoRA. This is also knows as  dynamic quantization

▶️ And there's some other formats like AWQ: Activation-aware Weight Quantization - which is a quantization method similar to GPTQ. There are several differences between AWQ and GPTQ as methods but the most important one is that AWQ assumes that not all weights are equally important for an LLM’s performance. For AWQ, best to use the vLLM package


##########################################

## If you just started playing with Quantized LLMs on local machine and are confused about which model format to download.

First noting that the same model is available in many formats and quantizations.

📌 You'll see things like Q4, 4bpw, GGUF, EXL2, and a few others.

📌 These are put into the name for easy identification and represent the technique used to quantize them.

📌 The Q and bpw numbers stand for how much a model has been compressed. Typically, q4, or 4bpw, is considered the sweet spot between quality and size. Smaller than that, and the models rapidly degrade.

📌 GGUF, EXL2, AWQ, GPTQ, and others refer to the format.

The format of the model will depend on the hardware you have.

📌 EXL2 is the best option if you have enough VRAM to load the entire model onto your GPU. But, it only works if the whole thing will fit. The backend, Exllama, is not able to use your system ram.

📌 GGUF is the best if the model does not fit inside the VRAM. Llama.cpp, which uses the GGUF format, is able to use your CPU, system ram, GPU, and VRAM all at the same time. This will allow you to load much larger models by splitting it between your GPU and CPU, but at the cost of speed.

But always be ready to fix issues. For example sometime some model, may be incredibly inference stack or hardware dependent for some strange issues.

Recently e.g. few people were facing issue with "Mixtral Hermes" model. There was a hint that "Mixtral Hermes" vocab size being 32002 instead of mixtral's standard one of 32000 played a role, as that was the cause of Together's initial issue (which they fixed).

----------

### Quantization with GGML

The way GGML quantizes weights is not as sophisticated as GPTQ’s. Basically, it groups blocks of values and rounds them to a lower precision. Some techniques, implement a higher precision for critical layers. In this case, every weight is stored in 4-bit precision, with the exception of half of the attention.wv and feed_forward.w2 tensors. Experimentally, this mixed precision proves to be a good tradeoff between accuracy and resource usage.

----------------------

Look at the source code of ggml.c file, we can see how the blocks are defined. For example, the block_q4_0 structure is defined as:

```c
#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
```

In GGML, weights are processed in blocks, each consisting of 32 values. For each block, a scale factor (delta) is derived from the largest weight value. All weights in the block are then scaled, quantized, and packed efficiently for storage (nibbles). This approach significantly reduces the storage requirements while allowing for a relatively simple and deterministic conversion between the original and quantized weights.

----------------------

## AWQ: Activation-aware Weight Quantization (aka static quantization)

AWQ is a relatively new format on the block which is a quantization method similar to GPTQ. There are many differences between AWQ and GPTQ as methods but the crucial one is that AWQ assumes that not all weights are equally important for an LLM’s performance.

So with AWQ, there is a small fraction of weights that will be skipped during quantization which helps with the quantization loss.

As a result, we get a significant speed-up compared to GPTQ whilst keeping similar performance.

For AWQ, its better to use vLLM package as its well-supported with vLLM