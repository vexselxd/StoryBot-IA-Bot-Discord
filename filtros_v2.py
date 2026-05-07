"""
filtros_v2.py - Pipeline mejorada de StoryBot v2

Filtros implementados:
[1] Validación de cierre - detecta y maneja frases que terminan en palabras vacías
[ ] Concordancia dinámica (siguiente)
[ ] Corte semántico mejorado (siguiente)
[ ] Longitud adaptativa (siguiente)
"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==================================================
# UTILIDADES BASE (reusadas de v1)
# ==================================================

PALABRAS_FUERTES = {
    # Originales de v1
    "muerte", "vida", "libertad", "alma", "sombra", "silencio",
    "fuego", "dolor", "noche", "verdad", "sangre", "luz", "infierno",
    # Ampliacion v2
    "destino", "olvido", "eternidad", "abismo", "ceniza", "ruina",
    "cielo", "tierra",
}

def softmax_temperature(probs, temperature=0.8, eps=1e-8):
    if temperature <= 0:
        temperature = 1e-6
    logits = np.log(probs + eps) / float(temperature)
    logits -= np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def sample_from_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ==================================================
# FILTRO 1 — VALIDACIÓN DE CIERRE
# ==================================================
# Detecta frases que terminan en palabras vacías (preposiciones,
# conjunciones, artículos, pronombres átonos) y las rechaza o
# las corta para tener un cierre válido.

PALABRAS_VACIAS_FINAL = {
    # Preposiciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por",
    "según", "sin", "sobre", "tras",
    # Conjunciones
    "y", "e", "o", "u", "pero", "mas", "sino", "aunque", "porque",
    "pues", "que", "si", "como", "cuando", "mientras", "donde",
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "lo", "del", "al",
    # Pronombres átonos / clíticos
    "me", "te", "se", "nos", "os", "le", "les",
    # Adverbios débiles para cerrar
    "muy", "más", "menos", "tan", "ya", "aún", "también",
    "casi", "solo", "sólo",
    # Demostrativos solos (terminar en "este", "ese" sin sustantivo)
    "este", "ese", "aquel", "esta", "esa", "aquella",
    "estos", "esos", "aquellos", "estas", "esas", "aquellas",
    # Posesivos átonos
    "mi", "tu", "su", "mis", "tus", "sus",
    "nuestro", "vuestra", "nuestra", "vuestro",
}


def termina_en_palabra_vacia(frase):
    """
    Devuelve True si la frase termina en una palabra vacía
    (preposición, conjunción, artículo, pronombre átono, etc.).
    """
    palabras = frase.strip().split()
    if not palabras:
        return False
    ultima = palabras[-1].lower().strip(".,;:¡!¿?\"'")
    return ultima in PALABRAS_VACIAS_FINAL




# ==================================================
# FUNCIONES HEREDADAS DE V1 (sin cambios)
# ==================================================

def finalizar_con_punto(frase):
    frase = frase.strip()
    if not frase:
        return frase
    if frase[-1] not in ".?!¡¿":
        frase += "."
    if frase[0].isalpha():
        frase = frase[0].upper() + frase[1:]
    return frase


def generar_texto(seed_text, next_words, model, tokenizer, max_sequence_len, temperature=0.8):
    result = seed_text.strip()
    index_word = getattr(tokenizer, "index_word", None)
    if index_word is None or not isinstance(index_word, dict):
        index_word = {idx: w for w, idx in tokenizer.word_index.items()}

    for _ in range(int(next_words)):
        seq = tokenizer.texts_to_sequences([result])[0]
        seq = pad_sequences([seq], maxlen=max_sequence_len - 1, padding="pre")
        preds = model.predict(seq, verbose=0)
        if preds.ndim == 2:
            preds = preds[0]
        probs = softmax_temperature(preds, temperature=temperature)
        idx = sample_from_probs(probs)
        word = index_word.get(idx)
        if not word:
            argmax_idx = int(np.argmax(probs))
            word = index_word.get(argmax_idx)
            if not word:
                break
        result = (result + " " + word).strip()

    return result


# ==================================================
# PIPELINE V2 - MODOS COMPARABLES
# ==================================================

def pipeline_v2_estricto(frase_cruda):
    """
    Modo ESTRICTO: si la frase termina en palabra vacía, la rechaza.
    Devuelve (frase_final, fue_aceptada).
    """
    if termina_en_palabra_vacia(frase_cruda):
        return frase_cruda, False
    return finalizar_con_punto(frase_cruda), True


def cortar_hasta_cierre_valido(frase, min_palabras=5):
    """
    Recorta la frase desde el final hacia atrás eliminando palabras vacías.
    Devuelve la frase recortada como string, o None si queda muy corta.
    """
    if not frase or not isinstance(frase, str):
        return None
    
    palabras = frase.strip().split()
    
    # Eliminar palabras vacías desde el final
    while palabras:
        ultima = palabras[-1].lower().strip(".,;:¡!¿?\"'")
        if ultima in PALABRAS_VACIAS_FINAL:
            palabras.pop()
        else:
            break
    
    if len(palabras) < min_palabras:
        return None
    
    return " ".join(palabras)


def pipeline_v2_permisivo(frase_cruda, min_palabras=5):
    """
    Modo PERMISIVO: corta palabras vacías del final hasta encontrar cierre válido.
    Garantiza siempre devolver una tupla (frase, bool).
    """
    if not frase_cruda or not isinstance(frase_cruda, str):
        return ("", False)
    
    if not termina_en_palabra_vacia(frase_cruda):
        return (finalizar_con_punto(frase_cruda), True)
    
    cortada = cortar_hasta_cierre_valido(frase_cruda, min_palabras)
    if cortada is None:
        return (frase_cruda, False)
    
    return (finalizar_con_punto(cortada), True)

def pipeline_v2_combinado(frase_cruda, min_palabras=5, min_pos_palabra_fuerte=6):
    """
    Pipeline combinado v2 (DEFAULT recomendado).
    
    Estrategia:
    1. Si hay palabra fuerte en posicion >= min_pos_palabra_fuerte, corta ahi.
       (Heredado de v1: "termina en palabra de impacto")
    2. Si no hay palabra fuerte, poda palabras vacias del final.
       (Nuevo en v2: evita cierres rotos)
    3. Si queda muy corta, descarta.
    
    Devuelve (frase_final, fue_aceptada).
    """
    if not frase_cruda or not isinstance(frase_cruda, str):
        return ("", False)
    
    palabras = frase_cruda.split()
    
    # PASO 1: Buscar palabra fuerte desde el final
    for i in range(len(palabras) - 1, -1, -1):
        token = palabras[i].lower().strip(".,;:¡!¿?\"'")
        if token in PALABRAS_FUERTES and i >= min_pos_palabra_fuerte:
            recortada = " ".join(palabras[:i + 1])
            return (finalizar_con_punto(recortada), True)
    
    # PASO 2: Si no hay palabra fuerte, podar vacias del final
    if termina_en_palabra_vacia(frase_cruda):
        cortada = cortar_hasta_cierre_valido(frase_cruda, min_palabras)
        if cortada is None:
            return (frase_cruda, False)
        return (finalizar_con_punto(cortada), True)
    
    # PASO 3: Frase OK como está
    return (finalizar_con_punto(frase_cruda), True)