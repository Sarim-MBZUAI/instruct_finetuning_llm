import os
import logging
from typing import Dict
import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import wandb
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

import os
import logging
from typing import Dict
import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from torch.utils.data import Dataset
import wandb
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Rest of the imports and code remain the same...

# Configuration
model_name_or_path = "upstage/TinySolar-248m-4k"  # Example model, replace with your model
dataset_name_or_path = "m-a-p/CodeFeedback-Filtered-Instruction"  # Replace with your dataset
output_dir = "./quantized_lora_fine_tuned_model_output"
wandb_project = "Quantized-Code-feedback-LoRA"
wandb_run_name = "quantized-lora-fine-tune-run-1"

# Dataset and tokenization settings
split_name = "train"
query_col_name = "query"
answer_col_name = "answer"
lang_col_name = "lang"
max_length = 512

# LoRA configuration
lora_r = 8
lora_alpha = 32
lora_dropout = 0.05

# Quantization configuration
load_in_4bit = True
bnb_4bit_compute_dtype = torch.bfloat16
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True

PROMPT_DICT = {
    "prompt": (
        "Below is a query in {lang}. Write a response that appropriately answers the query.\n\n"
        "### Query:\n{query}\n\n### Response:"
    ),
}

def format_prompt(example):
    return PROMPT_DICT["prompt"].format_map(example)

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = format_prompt(example)
        answer = example[answer_col_name]

        encoded = self.tokenizer.encode_plus(
            prompt + answer,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # Create labels: -100 for prompt tokens, actual token ids for answer tokens
        labels = input_ids.clone()
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class GradientLoggingCallback(transformers.TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.logging_steps == 0:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}, step=state.global_step)
                    wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())}, step=state.global_step)

def main():
    wandb.init(project=wandb_project, name=wandb_run_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"]  # Adjust based on your model architecture
    )

    # Wrap the model with LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and process dataset
    dataset = datasets.load_dataset(dataset_name_or_path, split=split_name)
    tokenized_dataset = TokenizedDataset(dataset, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        callbacks=[GradientLoggingCallback(model)]
    )

    trainer.train()

    # Save the LoRA model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()