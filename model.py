import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def softmax_temperature(probs, temperature=0.8, eps=1e-8):
    if temperature <= 0:
        temperature = 1e-6
    logits = np.log(probs + eps) / float(temperature)
    logits -= np.max(logits)
    exps = np.exp(logits)
    out = exps / np.sum(exps)
    return out

def sample_from_probs(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

PALABRAS_FUERTES = {
    "muerte", "vida", "libertad", "alma", "sombra", "silencio",
    "fuego", "dolor", "noche", "verdad", "sangre", "luz", "infierno",
}

def cortar_por_palabra_fuerte(frase, min_pos=6):
    palabras = frase.split()
    for i in range(len(palabras) - 1, -1, -1):
        token = palabras[i].lower().strip(".,;:¡!¿?")
        if token in PALABRAS_FUERTES and i >= min_pos:
            return " ".join(palabras[:i + 1])
    return frase

def finalizar_con_punto(frase):
    frase = frase.strip()
    if not frase:
        return frase
    if frase[-1] not in ".?!¡¿":
        frase += "."
    if frase[0].isalpha():
        frase = frase[0].upper() + frase[1:]
    return frase

def es_frase_valida(frase):
    f = frase.strip()
    palabras = f.split()

    if len(palabras) < 5:
        return False
    if f.count(" ") > 30:
        return False

    errores_articulos = [
        "la dios", "la alma", "la ruinas", "el sombra", "el guerra", "el soledad",
        "un sombra", "una fuego", "el tierra", "el noche", "el victoria",
        "un historia", "el almas", "un guerra", "un sangre"
    ]
    lower = f.lower()
    if any(e in lower for e in errores_articulos):
        return False

    problematicas = [
        "mí me", "me mí", "con mí",
        "los luz", "los fuego", "los mundo", "los espejo",
        "te mí", "mí te", "me niega las", "los primer será", "yo me negra",
    ]
    if any(p in lower for p in problematicas):
        return False

    if " un que" in lower or " una que" in lower:
        return False

    for i in range(len(palabras) - 1):
        if palabras[i].lower() == palabras[i + 1].lower():
            return False

    verbos = {"es", "fue", "arde", "vive", "muere", "existe",
              "quema", "llora", "habla", "calla", "resiste",
              "crece", "cae", "perdura", "siente"}
    if not any(v in lower for v in verbos):
        return False

    return True

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
