# StoryBot · AI Writing Assistant for Discord

> An experiment in minimalist AI design: ~1,500 training phrases, a small neural network, and seven hand-written filters. The question was whether a specific writing style — philosophical Spanish prose — could be captured without massive datasets, expensive GPUs, or retraining cycles.

---

## The thesis

Most ML projects scale by adding more: more data, more parameters, more compute. This one goes the other direction. The model is intentionally small. The dataset is intentionally limited. The intelligence lives **around** the model, not inside it — in a pipeline of post-processing filters that catch and reshape the output until it reads like the source material.

Send `$ideas` followed by any phrase and the bot completes it:

```
User:    $ideas El alma paga sola sus pecados
Bot:  ▶  El alma paga sola sus pecados del alma.

User:    $ideas La existencia se basa en una sola cosa
Bot:  ▶  La existencia se basa en una sola cosa con cicatrices más en mí
         pero eso me reconozco sentir olvidado no confundas eso con libertad.
```

---

## Stack

![Python](https://img.shields.io/badge/Python-0a0a2e?style=for-the-badge&logo=python&logoColor=f5c842)
![TensorFlow](https://img.shields.io/badge/TensorFlow-0a0a2e?style=for-the-badge&logo=tensorflow&logoColor=f5c842)
![Keras](https://img.shields.io/badge/Keras-0a0a2e?style=for-the-badge&logo=keras&logoColor=f5c842)
![discord.py](https://img.shields.io/badge/discord.py-0a0a2e?style=for-the-badge&logo=discord&logoColor=f5c842)
![Google Colab](https://img.shields.io/badge/Google_Colab-0a0a2e?style=for-the-badge&logo=googlecolab&logoColor=f5c842)
![JSON](https://img.shields.io/badge/JSON-0a0a2e?style=for-the-badge&logo=json&logoColor=f5c842)

---

## Why philosophical Spanish

The choice wasn't accidental. Philosophical prose has dense vocabulary, recurring imagery, and a recognizable rhythm — exactly the kind of texture a small model can learn to imitate when general fluency is out of reach. Words like *alma*, *silencio*, *fuego*, *muerte* carry weight on their own, so even imperfect grammar reads as deliberate when the tone holds.

This decision shaped everything else: the dataset size, the filter design, even which words the bot prefers to end sentences on.

---

## How it was built

### 1. Dataset by design, not by limitation

~1,500 short philosophical phrases in Spanish. The small size was a deliberate constraint — the goal was to test whether a tight, curated corpus could capture a writing style more cleanly than a massive general-purpose one. Less noise, more voice.

### 2. A small model that knows its lane

An n-gram-based neural network trained with TensorFlow/Keras. It learns which words tend to follow which sequences, then samples completions token by token using temperature-controlled softmax.

- Training environment: Google Colab (free GPU)
- Exported as: `red_neuronal.h5`
- Tokenizer serialized to: `tokenizer.json`

The model alone is not enough to produce clean output. That's the point.

### 3. Seven filters that do the heavy lifting

Instead of retraining when the model produced bad sentences, every error pattern observed in real usage was answered with a hand-written filter. The pipeline is the product:

| # | Filter | What it catches |
|---|--------|-----------------|
| 1 | **Length validation** | Discards outputs under 5 words or over 30 — too short to mean anything, too long to stay coherent |
| 2 | **Article concordance** | Catches grammatical mismatches like *la dios*, *el sombra*, *un guerra* |
| 3 | **Pronoun & syntax errors** | Filters known broken patterns like *mí me*, *los luz*, *te mí* |
| 4 | **Consecutive repetition** | Rejects sentences where the same word appears twice in a row |
| 5 | **Verb requirement** | Demands at least one verb from a curated list — no verb, no sentence |
| 6 | **Strong-word cutoff** | Cuts the sentence at the last semantically heavy word (*muerte*, *fuego*, *silencio*…) so it ends on impact |
| 7 | **Cleanup pass** | Capitalizes the first letter and ensures the sentence closes with punctuation |

The loop was simple: **observe error → write filter → test → repeat.** No retraining, no extra data, no GPU time.

### 4. The deployment problem (and what it taught me)

After training in Colab and moving the model to the local environment, the bot broke. Two libraries required conflicting versions of the same dependency, and version pinning couldn't resolve it.

The fix wasn't to fight the dependency tree. It was to **remove the dependency from the inference path entirely** — by serializing the tokenizer to JSON and loading the vocabulary directly. The library that was causing the conflict was no longer needed at runtime.

That shift, from "make the conflict work" to "make the conflict irrelevant," is the lesson I keep coming back to.

---

## Project structure

```
StoryBot-IA-Bot-Discord/
├── main.py              # Discord bot + inference logic
├── model.py             # Generation pipeline + filters
├── red_neuronal.h5      # Trained model weights
├── tokenizer.json       # Serialized tokenizer vocabulary
├── max_sequence_len.pkl # Max sequence length for padding
└── requirements.txt     # Pinned dependencies
```

---

## What I learned

- **The cleanest fix to a dependency conflict is often to remove the dependency, not pin it.**
- **Filters can replace retraining** when errors are predictable and the dataset is small. Cheaper, faster, and the failure modes stay legible.
- **Style is easier to capture than meaning.** A tight thematic dataset gives a small model a voice it could never achieve trying to be general.

---

## Limitations

The grammar isn't always perfect, and the bot is bounded by the vocabulary of its training set — it won't surprise you with words it never saw. The filters can reject too aggressively when temperature is high. These are tradeoffs of the design, not bugs to fix.

---

## Author

![Portfolio](https://img.shields.io/badge/Portfolio-0a0a2e?style=for-the-badge&logoColor=f5c842&label=✦)
![GitHub](https://img.shields.io/badge/GitHub-0a0a2e?style=for-the-badge&logo=github&logoColor=f5c842)

**Christopher** · [vexsel.pythonanywhere.com](https://vexsel.pythonanywhere.com) · [github.com/vexselxd](https://github.com/vexselxd)
