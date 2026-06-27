import argparse
import os
import json
import re
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


def limpar_texto(texto):
    texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'#{1,6}\s', '', texto)
    texto = re.sub(r'\*\*|__|\*|_|`{1,3}', '', texto)
    texto = re.sub(r'\|[-\s|]+\|', '', texto)
    texto = re.sub(r'\|.*?\|', ' ', texto)
    texto = re.sub(r'^\d+\s*$', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'http\S+', '', texto)
    linhas = [l.strip() for l in texto.split('\n')]
    linhas = [l for l in linhas if len(l) > 20]
    texto = '\n'.join(linhas)
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    texto = re.sub(r' {2,}', ' ', texto)
    return texto.strip()


def texto_valido(texto):
    if len(texto) < 200:
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
        dataset = load_dataset("gutoportelaa/DOMPI-2025", split=split)
        count = 0
        for row in dataset:
            if len(exemplos) >= MAX_EXEMPLOS:
                break
            texto = row.get("texto", "").strip()
            texto = limpar_texto(texto)
            if texto_valido(texto):
                if len(texto) > 6000:
                    texto = texto[:6000]
                exemplos.append({"text": texto})
                count += 1
            else:
                descartados += 1
        print(f"  → {count} exemplos aproveitados")

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

    print(f"\nDescartados (ruído/curtos): {descartados}")
    print(f"Treino: {len(treino)} exemplos → train.jsonl")
    print(f"Teste:  {len(teste)} exemplos  → test.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dompi", type=str)
    args = parser.parse_args()
    main(args)