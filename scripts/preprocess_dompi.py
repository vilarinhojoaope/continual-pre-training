# scripts/preprocess_dompi.py
import argparse
import os
import json
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

def main(args):
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "processed.jsonl")
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for split in SPLITS:
            print(f"Carregando split: {split}...")
            dataset = load_dataset("gutoportelaa/DOMPI-2025", split=split)
            count = 0
            for row in dataset:
                texto = row.get("texto", "").strip()
                if texto:
                    f.write(json.dumps({"text": texto}, ensure_ascii=False) + "\n")
                    count += 1
            print(f"  → {count} exemplos")
            total += count

    print(f"\nTotal salvo em {out_path}: {total} exemplos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dompi", type=str)
    args = parser.parse_args()
    main(args)