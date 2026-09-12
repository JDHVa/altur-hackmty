# -*- coding: utf-8 -*-
import os
import io
import sys
import asyncio
import subprocess

import numpy as np
import soundfile as sf
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

EDGE_VOICES = [
    "es-ES-AlvaroNeural",
    "es-ES-ElviraNeural",
    "es-AR-ElenaNeural",
    "es-CO-GonzaloNeural",
    "es-US-PalomaNeural",
]

GTTS_VARIANTS = [("es", "com.mx"), ("es", "es")]

OUTPUT_DIR = "datasets_externos/Synthetic_Multi_8kHz"
CODEC = "g711_ulaw"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def mp3_to_array(mp3_bytes):
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                       input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[:200])
    wave, sr = sf.read(io.BytesIO(p.stdout))
    return (wave[:, 0] if wave.ndim > 1 else wave).astype(np.float32), sr


def save_processed(wave, sr, name):
    proc, out_sr = augment_chain(wave, sr, codec=CODEC)
    path = os.path.join(OUTPUT_DIR, name)
    sf.write(path, proc, out_sr, subtype="PCM_16")
    return path, out_sr, len(proc) / out_sr


async def gen_edge():
    import edge_tts
    n = 0
    for voice in EDGE_VOICES:
        for i, text in enumerate(PHRASES):
            comm = edge_tts.Communicate(text, voice)
            mp3 = b""
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    mp3 += ch["data"]
            wave, sr = mp3_to_array(mp3)
            path, osr, dur = save_processed(wave, sr, f"edge_{voice}_{i:02d}.wav")
            n += 1
            print(f" [edge] {os.path.basename(path)} ({dur:.1f}s)")
    return n


def gen_gtts():
    from gtts import gTTS
    n = 0
    for lang, tld in GTTS_VARIANTS:
        for i, text in enumerate(PHRASES):
            buf = io.BytesIO()
            gTTS(text, lang=lang, tld=tld).write_to_fp(buf)
            wave, sr = mp3_to_array(buf.getvalue())
            path, osr, dur = save_processed(wave, sr, f"gtts_{tld}_{i:02d}.wav")
            n += 1
            print(f" [gtts] {os.path.basename(path)} ({dur:.1f}s)")
    return n


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Sinteticos multi-motor (Edge + gTTS), canal g711_ulaw ===")
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    n_edge = asyncio.run(gen_edge())
    n_gtts = gen_gtts()
    print(f"\nTotal: {n_edge} edge + {n_gtts} gtts = {n_edge + n_gtts} en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
