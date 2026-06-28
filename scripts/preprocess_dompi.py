import argparse
import os
import json
import random
from datasets import load_dataset

SPLITS = [
    'cocais', 'mangabeiras', 'serra_da_capivara', 'vale_do_rio_guaribas',
    'chapada_vale_do_rio_itaim', 'carnaubais', 'tabuleiros_alto_parnaiba',
    'vale_dos_rios_piaui_e_itaueiras', 'vale_do_caninde', 'planice_litoran',
    'entre_rios', 'vale_do_sambito',
]

MAX_EXEMPLOS = 35000
TRAIN_RATIO  = 0.8
SEED         = 42


def texto_valido(texto):
    if not texto or len(texto) < 300:
        return False
    return True


def main(args):
    os.makedirs(args.output, exist_ok=True)
    exemplos = []
    descartados = 0

    for split in SPLITS:
        if len(exemplos) >= MAX_EXEMPLOS:
            break
        print(f"Carregando split: {split}...")
        dataset = load_dataset("pcarvalhomgs/DOMPI-2025", split=split)
        count = 0
        for row in dataset:
            if len(exemplos) >= MAX_EXEMPLOS:
                break

            # Usa texto_treino — já pré-processado pelo autor do dataset
            texto = (row.get("texto_treino") or "").strip()

            if texto_valido(texto):
                if len(texto) > 6000:
                    texto = texto[:6000]
                exemplos.append({"text": texto})
                count += 1
            else:
                descartados += 1

        print(f"  → {count} aproveitados")

    random.seed(SEED)
    random.shuffle(exemplos)

    corte  = int(len(exemplos) * TRAIN_RATIO)
    treino = exemplos[:corte]
    teste  = exemplos[corte:]

    train_path = os.path.join(args.output, "train.jsonl")
    test_path  = os.path.join(args.output, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in treino:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for ex in teste:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDescartados: {descartados}")
    print(f"Treino: {len(treino)} exemplos → train.jsonl")
    print(f"Teste:  {len(teste)} exemplos  → test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dompi", type=str)
    args = parser.parse_args()
    main(args)