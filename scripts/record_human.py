import os
import sys
import argparse

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features.telephony_aug import augment_chain

PROMPTS = [
    "Hola, quiero consultar el saldo de mi tarjeta de credito.",
    "No reconozco un cargo en mi estado de cuenta, quiero aclararlo.",
    "Necesito reportar mi tarjeta como robada, por favor.",
    "Se me olvido el NIP del cajero, como lo recupero?",
    "Quiero saber los requisitos para un prestamo personal.",
    "Mi transferencia no ha pasado, me pueden ayudar?",
    "Si, confirmo que yo hice esa compra ayer en la noche.",
    "Me puede comunicar con un ejecutivo, por favor?",
    "Habla libremente unos segundos sobre tu dia (improvisado).",
    "Cuenta que hiciste el fin de semana (improvisado).",
]


def record_clip(seconds, sr):
    import sounddevice as sd
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speaker', required=True)
    ap.add_argument('--clips', type=int, default=10)
    ap.add_argument('--seconds', type=float, default=12.0)
    ap.add_argument('--sr', type=int, default=16000)
    ap.add_argument('--codec', default='g711_ulaw')
    ap.add_argument('--out', default='datasets_externos/Human_MX_8kHz')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Grabando {args.clips} clips de {args.seconds}s para hablante '{args.speaker}'.")
    print("Lee cada frase en voz natural (con muletillas y pausas si quieres).\n")

    saved = 0
    for i in range(args.clips):
        prompt = PROMPTS[i % len(PROMPTS)]
        input(f"[{i+1}/{args.clips}] Enter para grabar -> \"{prompt}\"")
        wave = record_clip(args.seconds, args.sr)
        proc, out_sr = augment_chain(wave, args.sr, codec=args.codec)
        path = os.path.join(args.out, f"human_{args.speaker}_{i:02d}.wav")
        sf.write(path, proc, out_sr, subtype='PCM_16')
        saved += 1
        print(f"  guardado: {path}  ({out_sr} Hz, {len(proc)/out_sr:.1f}s)\n")

    print(f"Listo. {saved} audios 'human' en {args.out} (mismo canal 8kHz que los sinteticos).")


if __name__ == '__main__':
    main()
