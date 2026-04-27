# StoryBot · AI Writing Assistant for Discord

> A Discord bot that completes phrases using a neural network trained from scratch on philosophical Spanish text.

---

## What it does

Send `$ideas` followed by any phrase and the bot completes it — generating text that follows the tone, vocabulary, and rhythm of philosophical writing.

```
User:    $ideas El alma paga sola sus pecados
Bot:  ▶  El alma paga sola sus pecados del alma.

User:    $ideas La existencia se basa en una sola cosa
Bot:  ▶  La existencia se basa en una sola cosa con cicatrices más en mí
         pero eso me reconozco sentir olvidado no confundas eso con libertad.
```

The grammar isn't always perfect — but the tone is consistent. That's what ~1,500 training sentences gets you.

---

## Stack

![Python](https://img.shields.io/badge/Python-0a0a2e?style=for-the-badge&logo=python&logoColor=f5c842)
![TensorFlow](https://img.shields.io/badge/TensorFlow-0a0a2e?style=for-the-badge&logo=tensorflow&logoColor=f5c842)
![Keras](https://img.shields.io/badge/Keras-0a0a2e?style=for-the-badge&logo=keras&logoColor=f5c842)
![discord.py](https://img.shields.io/badge/discord.py-0a0a2e?style=for-the-badge&logo=discord&logoColor=f5c842)
![Google Colab](https://img.shields.io/badge/Google_Colab-0a0a2e?style=for-the-badge&logo=googlecolab&logoColor=f5c842)
![JSON](https://img.shields.io/badge/JSON-0a0a2e?style=for-the-badge&logo=json&logoColor=f5c842)

---

## How it was built

### 1. Dataset
~1,500 short philosophical phrases in Spanish. Small by design — the goal was to test whether a minimal dataset could capture a specific writing style, not to build a general-purpose language model.

### 2. Model
Trained using **n-grams** with TensorFlow/Keras. The model learns which words tend to follow which sequences, then generates completions token by token.

- Architecture: n-gram based neural network
- Framework: TensorFlow / Keras
- Training environment: Google Colab (free GPU)
- Exported as: `red_neuronal.h5`

### 3. The dependency conflict problem
After training in Colab and exporting to local environment, the bot broke — two libraries required conflicting versions of the same dependency. Neither worked with the other installed.

**The fix:** serialize the tokenizer to JSON (`tokenizer.json`) instead of relying on the library to reconstruct it at runtime. This removed the problematic dependency from the inference pipeline entirely — the model loads the vocabulary directly from the file and runs cleanly regardless of what versions are installed.

This is what made deployment actually work.

### 4. Output filters
The raw model output had consistent error patterns. Instead of retraining (expensive in RAM and time), 7 post-processing filters were built by hand — each one targeting a specific type of bad output observed in real usage.

```
Observe error → write filter → test → repeat
```

This kept RAM usage low and avoided the need for a larger dataset.

### 5. Deployment
- Model trained in Google Colab
- Exported to local environment
- Integrated with Discord via `discord.py`
- Tokenizer serialized to JSON to resolve version conflicts
- Run inside a virtual environment (`.venv`)

---

## Project structure

```
StoryBot-IA-Bot-Discord/
├── main.py              # Discord bot + inference logic
├── model.py             # Model architecture
├── red_neuronal.h5      # Trained model weights
├── tokenizer.json       # Serialized tokenizer vocabulary
├── max_sequence_len.pkl # Max sequence length for padding
└── requirements.txt     # Pinned dependencies
```

---

## What I learned

- Training and exporting ML models for production is a different problem than training them in a notebook
- Dependency conflicts in ML environments are not always solvable by pinning versions — sometimes the cleaner solution is to remove the dependency from the critical path entirely
- Post-processing filters are a practical alternative to retraining when the dataset is small and errors are predictable
- Serializing model artifacts (tokenizer, sequence length) is essential for reproducible inference across environments

---

## Author

![Portfolio](https://img.shields.io/badge/Portfolio-0a0a2e?style=for-the-badge&logoColor=f5c842&label=✦)
![GitHub](https://img.shields.io/badge/GitHub-0a0a2e?style=for-the-badge&logo=github&logoColor=f5c842)

**Christopher** · [vexsel.pythonanywhere.com](https://vexsel.pythonanywhere.com) · [github.com/vexselxd](https://github.com/vexselxd)
