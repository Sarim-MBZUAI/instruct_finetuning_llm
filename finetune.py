import os
import logging
from typing import Dict
import torch
import datasets
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import wandb

# Configuration
model_name_or_path = "upstage/TinySolar-248m-4k"
dataset_name_or_path = "sahil2801/CodeAlpaca-20k"
output_dir = "./fine_tuned_tinysolar_codealpaca"
wandb_project = "tinysolar-codealpaca-finetune"
wandb_run_name = "fine-tune-run-1"

# Dataset and tokenization settings
split_name = "train"
output_col_name = "output"
inst_col_name = "instruction"
input_col_name = "input"
max_length = 512

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_prompt(example):
    if example.get(input_col_name, "").strip():
        return PROMPT_DICT["prompt_input"].format_map(example)
    else:
        return PROMPT_DICT["prompt_no_input"].format_map(example)

class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        prompt = format_prompt(example)
        output = example[output_col_name]

        encoded = self.tokenizer.encode_plus(
            prompt + output,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()

        # Create labels: -100 for prompt tokens, actual token ids for output tokens
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

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Load and process dataset
    dataset = datasets.load_dataset(dataset_name_or_path, split=split_name)
    tokenized_dataset = TokenizedDataset(dataset, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        report_to="wandb",
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        fp16=False,  # Disable mixed precision training
        bf16=True,   # Enable bfloat16 training
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


    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    wandb.finish()

if __name__ == "__main__":
    main()