# -*- coding: utf-8 -*-
import os
import sys
import argparse

os.environ['COQUI_TOS_AGREED'] = '1'

DEFAULT_TEXT = (
    "Hola, le hablo del departamento de prevencion de fraudes de su banco. "
    "Detectamos un cargo sospechoso en su cuenta y necesito confirmar su identidad. "
    "Por favor, digame el codigo que acaba de recibir por mensaje para poder proteger su dinero."
)


def main():
    ap = argparse.ArgumentParser(description='Clona una voz con XTTS-v2 (uso defensivo: probar el detector)')
    ap.add_argument('--ref', required=True, help='wav de referencia de la voz a clonar (~6-15s, limpio)')
    ap.add_argument('--text', default=DEFAULT_TEXT)
    ap.add_argument('--language', default='es')
    ap.add_argument('--out', default='voz_clonada.wav')
    args = ap.parse_args()

    if not os.path.exists(args.ref):
        print('No existe el wav de referencia:', args.ref)
        return

    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})
    from TTS.api import TTS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Cargando XTTS-v2 en {device}...')
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
    tts.tts_to_file(text=args.text, speaker_wav=args.ref, language=args.language, file_path=args.out)
    print(f'Voz clonada -> {args.out}')
    print(f'Pruebala en el detector:  .venv\\Scripts\\python.exe scripts/analyze_audio.py {args.out}')


if __name__ == '__main__':
    main()
