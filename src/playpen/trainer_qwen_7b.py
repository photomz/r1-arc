#!/usr/bin/env python
# coding: utf-8

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
import time

import sys, os

sys.path.append("../..")
from src import training

os.environ["WANDB_PROJECT"] = "r1-arc"
dataset = training.load_dataset("photonmz/arc_plain")

N_TRAJECTORIES = 4
VERSION = 1
max_prompt_length = 14000
max_completion_length = 15000
max_seq_length = 30000  # Can increase for longer think traces
lora_rank = 64  # Larger rank = smarter, but slower

assert max_prompt_length + max_completion_length < max_seq_length
# what % of GRAM is allocated to vllm+model. Not fixed, easily OOM near boundary.
gram_util = 0.60  # <--.75 <- .9

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=gram_util,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Remove QKVO if out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

training_args = GRPOConfig(
    use_vllm=True,  # use vLLM for fast inference!
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=N_TRAJECTORIES * 1,
    # I think increasing grad_accum_steps multiplies KV Cache size (not offloaded well)
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=N_TRAJECTORIES,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs=10,  # Set to 1 for a full training run
    max_steps=400 * 10,
    save_steps=200,
    # num_iterations = 10,
    max_grad_norm=0.1,
    report_to="wandb",  # Can use Weights & Biases
    output_dir="outputs",
    run_name=f'v{VERSION}-{time.strftime("%Y%m%d_%H%M")}',
    log_completions=True,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=training.REWARD_FNS,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()

print("Wow, didn't crash!")
