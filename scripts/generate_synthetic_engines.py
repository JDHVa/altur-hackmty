# -*- coding: utf-8 -*-
import os
import io
import sys
import wave
import argparse
import tempfile
import numpy as np
import soundfile as sf

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
    "Buenos días, necesito activar mi tarjeta nueva.",
    "¿Puedo aumentar el límite de mi crédito este mes?",
    "Tengo una duda sobre los intereses de mi cuenta.",
    "Quiero domiciliar el pago de mi recibo de luz.",
    "Perdí mi teléfono y quiero bloquear la aplicación.",
    "¿Cómo puedo abrir una cuenta de ahorro para mi hijo?",
    "Me cobraron una anualidad que no autoricé.",
    "Necesito una carta de saldo para un trámite.",
    "¿Cuál es el tipo de cambio del dólar hoy?",
    "Quiero reportar un movimiento sospechoso en mi cuenta.",
]

CODEC = "g711_ulaw"


def save_aug(wave_np, sr, out_dir, name):
    proc, out_sr = augment_chain(wave_np, sr, codec=CODEC)
    path = os.path.join(out_dir, name)
    sf.write(path, proc, out_sr, subtype="PCM_16")
    return path


def gen_sapi(out_dir):
    import pyttsx3
    os.makedirs(out_dir, exist_ok=True)
    eng = pyttsx3.init()
    for v in eng.getProperty('voices'):
        if 'es-MX' in str(getattr(v, 'languages', '')) or 'Sabina' in v.name:
            eng.setProperty('voice', v.id)
            break
    base_rate = eng.getProperty('rate')
    n = 0
    for rate in (base_rate, base_rate - 30):
        eng.setProperty('rate', rate)
        for i, text in enumerate(PHRASES):
            tmp = os.path.join(tempfile.gettempdir(), f'sapi_{rate}_{i}.wav')
            eng.save_to_file(text, tmp)
            eng.runAndWait()
            w, sr = sf.read(tmp)
            w = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
            save_aug(w, sr, out_dir, f'sapi_r{rate}_{i:02d}.wav')
            os.remove(tmp)
            n += 1
    print(f'SAPI: {n} clips en {out_dir}')
    return n


def gen_piper(out_dir):
    from piper import PiperVoice
    os.makedirs(out_dir, exist_ok=True)
    v = PiperVoice.load('models/piper/es_MX-ald-medium.onnx', 'models/piper/es_MX-ald-medium.onnx.json')
    n = 0
    for i, text in enumerate(PHRASES):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            v.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, 'rb') as wf:
            sr = wf.getframerate()
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        save_aug(data, sr, out_dir, f'piper_{i:02d}.wav')
        n += 1
    print(f'Piper: {n} clips en {out_dir}')
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['sapi', 'piper', 'both'], default='both')
    args = ap.parse_args()
    if args.engine in ('sapi', 'both'):
        gen_sapi('datasets_externos/Synthetic_SAPI_8kHz')
    if args.engine in ('piper', 'both'):
        gen_piper('datasets_externos/Synthetic_Piper_8kHz')


if __name__ == '__main__':
    main()
