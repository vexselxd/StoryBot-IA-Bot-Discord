import os
import json
import pickle
import discord
from discord.ext import commands
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TOKEN = os.getenv("DISCORD_TOKEN") 
MODEL_PATH = "./red_neuronal.h5"
TOKENIZER_JSON_PATH = "./tokenizer.json"
MAXLEN_PATH = "./max_sequence_len.pkl"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="$", intents=intents)

from model import (
    generar_texto,
    es_frase_valida,
    cortar_por_palabra_fuerte,
    finalizar_con_punto,
)

_model = None
_tokenizer = None
_max_sequence_len = None

def cargar_recursos():
    global _model, _tokenizer, _max_sequence_len
    if _model is None:
        _model = load_model(MODEL_PATH)
        with open(TOKENIZER_JSON_PATH, "r", encoding="utf-8") as f:
            _tokenizer = tokenizer_from_json(f.read())
        with open(MAXLEN_PATH, "rb") as f:
            _max_sequence_len = pickle.load(f)

@bot.event
async def on_ready():
    cargar_recursos()
    print(f"Bot conectado como {bot.user} Modelo y tokenizer cargados.")

@bot.command()
async def ideas(ctx, *, seed_text: str):
    cargar_recursos()
    frase_cruda = generar_texto(
        seed_text=seed_text,
        next_words=20,
        model=_model,
        tokenizer=_tokenizer,
        max_sequence_len=_max_sequence_len,
        temperature=0.8
    )
    if es_frase_valida(frase_cruda):
        frase_corta = cortar_por_palabra_fuerte(frase_cruda)
        frase_final = finalizar_con_punto(frase_corta)
        await ctx.send(f"➤ {frase_final}")
    else:
        await ctx.send(f"✘ Descartada: {frase_cruda}")

if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("Falta la variable de entorno DISCORD_TOKEN")
    bot.run(TOKEN)



#$env:DISCORD_TOKEN = "Discord token"