# scripts/train_continual.py
import argparse
import yaml
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def load_jsonl(path, text_col="text"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({text_col: obj[text_col]})
    return Dataset.from_list(rows)

def main(args):
    cfg = load_config(args.config)

    print(f"Carregando modelo: {cfg['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,           # Auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    print(f"Carregando dados: {cfg['data_path']}")
    dataset = load_jsonl(cfg["data_path"], text_col=cfg["text_column"])

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        max_steps=cfg["max_steps"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        seed=cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=cfg["text_column"],
        max_seq_length=cfg["max_seq_length"],
        args=training_args,
    )

    print("Iniciando Continual Pre-Training...")
    trainer.train()
    print(f"Treinamento concluído. Checkpoints salvos em: {cfg['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cpt.yaml")
    args = parser.parse_args()
    main(args)