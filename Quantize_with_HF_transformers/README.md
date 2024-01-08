# About this script

A Python script to quantize GPT models with HuggingFace 'transformers' library.

------------------

# Usage

### Install all dependencies

```bash
pip install -r requirements.txt
```

### Run the script

```bash

python Quantize_with_HF_transformers.py --model_id 'mistralai/Mistral-7B-v0.1' --bits 4 --dataset 'wikitext2' --group_size 32 --device_map 'auto'

```

---------


# Features

- Quantizing at various GPTQ precisions (8bit and 4bit).

# Parameters:

- `model_id`: The model path/id from huggingface repository or local directory.

- `bits`:  The number of bits to quantize to, supported numbers are (2, 3, 4, 8).

- `dataset`:  The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper [‘wikitext2’,‘c4’,‘c4-new’,‘ptb’,‘ptb-new’]

- `group_size`: The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.

- `device_map`: Device mapping configuration for loading the model. Example: 'auto', 'cpu', 'cuda:0', etc. - Default `"auto"`


-------------

For all params of GPTQConfig check its official doc

https://huggingface.co/docs/transformers/main_classes/quantization#transformers.GPTQConfig



## License

Apache 2