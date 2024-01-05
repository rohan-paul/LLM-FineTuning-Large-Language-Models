import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft import prepare_model_for_kbit_training, get_peft_model
from transformers import GPTQConfig

from trl import SFTTrainer

# Fine-tunes Llama 2 model on Guanaco dataset using GPTQ and peft.

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    learning_rate: Optional[float] = field(default=0.0002, metadata={"help": 'The learning rate'})
    max_grad_norm: Optional[float] = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    weight_decay: Optional[int] = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64, metadata={"help": "Lora R dimension."})
    max_seq_length: Optional[int] = field(default=512)

    model_name: Optional[str] = field(
        default="TheBloke/Llama-2-7B-GPTQ",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

###########################################################

parser = HfArgumentParser(TrainingArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def prepare_lora_model(args):
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
        print("=" * 80)

    # Load the entire model on the GPU 0
    device_map = {"":0}
    # switch to `device_map = "auto"` for multi-GPU
    # device_map = "auto"


    # need to disable exllama kernel
    # exllama kernel are not very stable for training
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=device_map,
        quantization_config= GPTQConfig(bits=4, use_exllama=False)
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    #For fine-tuning llama2 models that have config.pretraining_tp>1 consider calling
    # model.config.pretraining_tp = 1
    model.config.pretraining_tp = 1

    lora_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, lora_config, tokenizer


training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

####################################################
model, lora_config, tokenizer = prepare_lora_model(script_args)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model.config.use_cache = False
dataset = load_dataset(script_args.dataset_name, split="train")

# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)

trainer.train()

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()