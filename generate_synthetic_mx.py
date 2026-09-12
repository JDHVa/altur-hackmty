import os
import io
import sys
import asyncio
import subprocess

import numpy as np
import soundfile as sf
import edge_tts
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features.telephony_aug import augment_chain

PHRASES = [
    "Hola, me gustaría consultar el saldo de mi tarjeta de crédito.",
    "No reconozco un cargo de quinientos pesos en mi estado de cuenta.",
    "Quiero reportar mi tarjeta de débito como robada.",
    "¿Cuáles son los requisitos para un préstamo personal?",
    "Mi transferencia SPEI no ha pasado, ¿pueden ayudarme?",
    "Necesito cancelar mi seguro de auto.",
    "Sí, confirmo que yo realicé esa compra ayer en la noche.",
    "¿Me puede comunicar con un ejecutivo por favor?",
    "Olvidé el NIP del cajero, ¿cómo lo recupero?",
    "Quisiera saber el estatus de mi aclaración.",
]

VOICES = ["es-MX-JorgeNeural", "es-MX-DaliaNeural"]
OUTPUT_DIR = "datasets_externos/Synthetic_MX_8kHz"
CODEC = "g711_ulaw"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def mp3_to_array(mp3_bytes):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                       input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:200])
    wave, sr = sf.read(io.BytesIO(p.stdout))
    return wave.astype(np.float32), sr


async def generate_audio():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Generando sinteticos es-MX (Edge-TTS) con canal identico a los humanos ===")
    count = 0
    for voice in VOICES:
        for i, text in enumerate(PHRASES):
            communicate = edge_tts.Communicate(text, voice)
            mp3 = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3 += chunk["data"]
            wave, sr = mp3_to_array(mp3)
            proc, out_sr = augment_chain(wave, sr, codec=CODEC)
            final_file = os.path.join(OUTPUT_DIR, f"synth_{voice}_{i:02d}.wav")
            sf.write(final_file, proc, out_sr, subtype="PCM_16")
            count += 1
            print(f" [OK] {final_file}  ({out_sr} Hz, {len(proc)/out_sr:.1f}s)")
    print(f"\nTotal sinteticos 8kHz (canal {CODEC}): {count}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(generate_audio())
