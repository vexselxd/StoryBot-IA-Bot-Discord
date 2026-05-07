"""
test_filtro1_v2.py - Comparacion ampliada: V1 vs V2 estricto/permisivo/combinado
sobre 15 seeds variadas.
"""
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Cargando recursos...")
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

modelo = load_model("./red_neuronal.h5")
with open("./tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())
with open("./max_sequence_len.pkl", "rb") as f:
    max_seq_len = pickle.load(f)

from model import (
    generar_texto as generar_v1,
    es_frase_valida as valida_v1,
    cortar_por_palabra_fuerte,
    finalizar_con_punto as final_v1,
)
from filtros_v2 import (
    pipeline_v2_estricto,
    pipeline_v2_permisivo,
    pipeline_v2_combinado,
    termina_en_palabra_vacia,
)

import numpy as np
np.random.seed(42)

seeds = [
    "El alma", "La noche", "El silencio", "La sangre", "El fuego",
    "La muerte", "Los dioses", "El destino", "La eternidad", "El abismo",
    "La libertad", "El olvido", "La sombra", "El cielo", "La verdad",
]

print("\n" + "=" * 75)
print("COMPARACION V1 vs V2 (estricto / permisivo / combinado)")
print("=" * 75)

stats = {"v1": 0, "v2_estricto": 0, "v2_permisivo": 0, "v2_combinado": 0}

for seed in seeds:
    print(f"\n┌─ Seed: '{seed}'")
    
    cruda = generar_v1(
        seed_text=seed,
        next_words=15,
        model=modelo,
        tokenizer=tokenizer,
        max_sequence_len=max_seq_len,
        temperature=0.8,
    )
    print(f"│  Cruda: {cruda}")
    
    # V1
    if valida_v1(cruda):
        v1_final = final_v1(cortar_por_palabra_fuerte(cruda))
        stats["v1"] += 1
        print(f"│  V1            ✓ {v1_final}")
    else:
        print(f"│  V1            ✗ descartada")
    
    # V2 estricto
    f, ok = pipeline_v2_estricto(cruda)
    if ok:
        stats["v2_estricto"] += 1
        print(f"│  V2 estricto   ✓ {f}")
    else:
        print(f"│  V2 estricto   ✗ descartada")
    
    # V2 permisivo
    f, ok = pipeline_v2_permisivo(cruda)
    if ok:
        stats["v2_permisivo"] += 1
        print(f"│  V2 permisivo  ✓ {f}")
    else:
        print(f"│  V2 permisivo  ✗ descartada")
    
    # V2 combinado (nuevo)
    f, ok = pipeline_v2_combinado(cruda)
    if ok:
        stats["v2_combinado"] += 1
        print(f"│  V2 combinado  ✓ {f}")
    else:
        print(f"│  V2 combinado  ✗ descartada")
    
    print(f"└─")

print("\n" + "=" * 75)
print("RESUMEN")
print("=" * 75)
total = len(seeds)
print(f"V1:            {stats['v1']}/{total}")
print(f"V2 estricto:   {stats['v2_estricto']}/{total}")
print(f"V2 permisivo:  {stats['v2_permisivo']}/{total}")
print(f"V2 combinado:  {stats['v2_combinado']}/{total}  <- DEFAULT propuesto")