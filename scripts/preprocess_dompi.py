# scripts/preprocess_dompi.py
import argparse
import os
import json
import random
from datasets import load_dataset

SPLITS = [
    'carnaubais',
    'tabuleiros_alto_parnaiba',
    'planice_litoran',
    'entre_rios',
    'vale_do_sambito',
    'vale_dos_rios_piaui_e_itaueiras',
    'cocais',
    'mangabeiras',
    'serra_da_capivara',
    'vale_do_rio_guaribas',
    'chapada_vale_do_rio_itaim',
    'vale_do_caninde',
]

MAX_EXEMPLOS = 3000
TRAIN_RATIO  = 0.8
SEED         = 42

def main(args):
    os.makedirs(args.output, exist_ok=True)
    exemplos = []

    # 1. Coleta os exemplos
    for split in SPLITS:
        if len(exemplos) >= MAX_EXEMPLOS:
            break
        print(f"Carregando split: {split}...")
        dataset = load_dataset("gutoportelaa/DOMPI-2025", split=split)
        count = 0
        for row in dataset:
            if len(exemplos) >= MAX_EXEMPLOS:
                break
            texto = row.get("texto", "").strip()
            if texto:
                exemplos.append({"text": texto})
                count += 1
        print(f"  → {count} exemplos")

    # 2. Embaralha de forma reproduzível
    random.seed(SEED)
    random.shuffle(exemplos)

    # 3. Divide treino/teste
    corte  = int(len(exemplos) * TRAIN_RATIO)
    treino = exemplos[:corte]
    teste  = exemplos[corte:]

    # 4. Salva os arquivos
    train_path = os.path.join(args.output, "train.jsonl")
    test_path  = os.path.join(args.output, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in treino:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in teste:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nTreino: {len(treino)} exemplos → train.jsonl")
    print(f"Teste:  {len(teste)} exemplos  → test.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dompi", type=str)
    args = parser.parse_args()
    main(args)