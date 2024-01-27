## Quantizing 🤗 Transformers models with the GPTQ method can be done in a only few lines after the Native support of GPTQ models in 🤗 Transformers

Note that for large model like 175B param, at least 4 GPU-hours will be needed if one uses a large dataset (e.g. `"c4"``).

And, many GPTQ models are already available on the Hugging Face Hub, which bypasses the need to quantize a model yourself in most use cases. Nevertheless, you can also quantize a model using your own dataset appropriate for the particular domain you are working on.

```py

# Quantizing 🤗 Transformers models with the GPTQ method

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                    quantization_config=quantization_config)

model.push_to_hub("username/Llama-2-7b-gptq-4bit")
tokenizer.push_to_hub("username/Llama-2-7b-gptq-4bit")

```

--------

The most important line is the one calling `GPTQConfig`:

Its first param bits (int) is the number of bits to quantize to, supported numbers are (2, 3, 4, 8).

`dataset`: The dataset used for calibration. I would leave “c4“ which seems to yield reasonable results. Other datasets are supported according to the documentation.

`tokenizer`: The tokenizer of Llama 2 7B that will be applied to c4.

-----------------

## There are two more key options that I did not use in the code below:

`desc_act` : Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly speed up inference but the perplexity may become slightly worse. If inference speed is not your concern, you should set desc_act to True.

`disable_exllama` : Whether to use exllama backend. Only works with bits = 4.

Now its `use_exllama` - bool, optional) — Whether to use exllama backend. Defaults to True if unset. Only works with bits = 4.

https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/quantization#transformers.GPTQConfig.use_exllama

True means that the support of exllama is set to False. By default, exllama is used. If you plan to use the model on a configuration with a small VRAM that will split the model to multiple devices with device_map, you should set disable_exllama to True.


