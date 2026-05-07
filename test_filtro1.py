"""
test_filtro1.py - Comparación: v1 vs v2_estricto vs v2_permisivo
sobre el mismo set de seeds.
"""
import os
import json
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

# Importar v1 y v2
from model import (
    generar_texto as generar_v1,
    es_frase_valida as valida_v1,
    cortar_por_palabra_fuerte,
    finalizar_con_punto as final_v1,
)
from filtros_v2 import (
    generar_texto as generar_v2,
    pipeline_v2_estricto,
    pipeline_v2_permisivo,
    termina_en_palabra_vacia,
)

# Para reproducibilidad relativa, fijamos seed numpy
import numpy as np
np.random.seed(42)

# Seeds variadas
seeds = [
    "El alma",
    "La noche",
    "El silencio",
    "La sangre",
    "El fuego",
    "La muerte",
    "Los dioses",
]

print("\n" + "=" * 70)
print("COMPARACIÓN V1 vs V2 (estricto y permisivo)")
print("=" * 70)

stats = {"v1_aceptadas": 0, "v2e_aceptadas": 0, "v2p_aceptadas": 0}

for seed in seeds:
    print(f"\n┌─ Seed: '{seed}'")
    
    # Generamos UNA frase cruda y la pasamos por los 3 pipelines
    cruda = generar_v1(
        seed_text=seed,
        next_words=15,  # bajamos a 15 (en v1 era 20)
        model=modelo,
        tokenizer=tokenizer,
        max_sequence_len=max_seq_len,
        temperature=0.8,
    )
    print(f"│  Cruda: {cruda}")
    print(f"│  Termina en palabra vacía: {termina_en_palabra_vacia(cruda)}")
    
    # === V1 (filtros originales) ===
    v1_ok = valida_v1(cruda)
    if v1_ok:
        v1_final = final_v1(cortar_por_palabra_fuerte(cruda))
        stats["v1_aceptadas"] += 1
        print(f"│  V1 ✓: {v1_final}")
    else:
        print(f"│  V1 ✗: descartada")
    
    # === V2 ESTRICTO ===
    v2e_final, v2e_ok = pipeline_v2_estricto(cruda)
    if v2e_ok:
        stats["v2e_aceptadas"] += 1
        print(f"│  V2 estricto ✓: {v2e_final}")
    else:
        print(f"│  V2 estricto ✗: descartada (termina en palabra vacía)")
    
    # === V2 PERMISIVO ===
    v2p_final, v2p_ok = pipeline_v2_permisivo(cruda)
    if v2p_ok:
        stats["v2p_aceptadas"] += 1
        print(f"│  V2 permisivo ✓: {v2p_final}")
    else:
        print(f"│  V2 permisivo ✗: descartada (muy corta tras podar)")
    
    print(f"└─")

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
total = len(seeds)
print(f"V1:           {stats['v1_aceptadas']}/{total} aceptadas")
print(f"V2 estricto:  {stats['v2e_aceptadas']}/{total} aceptadas")
print(f"V2 permisivo: {stats['v2p_aceptadas']}/{total} aceptadas")


# === V2 PERMISIVO ===
resultado = pipeline_v2_permisivo(cruda)
print(f"│  DEBUG resultado permisivo: {resultado!r}")
v2p_final, v2p_ok = resultado