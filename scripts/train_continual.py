# scripts/train_continual.py
import argparse
import yaml
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

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
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(cfg["seed"]),
    )

    print(f"Carregando dados: {cfg['data_path']}")
    dataset = load_jsonl(cfg["data_path"], text_col=cfg["text_column"])

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        max_steps=int(cfg["max_steps"]),
        warmup_steps=int(cfg["warmup_steps"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        fp16=bool(cfg.get("fp16", False)),
        bf16=bool(cfg.get("bf16", False)),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        seed=int(cfg["seed"]),
        report_to="none",
        dataset_text_field=cfg["text_column"],
        max_seq_length=int(cfg["max_seq_length"]),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
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