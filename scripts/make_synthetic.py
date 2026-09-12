# -*- coding: utf-8 -*-
import os
import sys
import asyncio
import argparse
import edge_tts

DEFAULT_TEXT = (
    "Hola, buenas tardes. Le llamo porque estaba revisando el estado de cuenta "
    "de mi tarjeta de credito y vi un cargo de tres mil pesos que yo no reconozco. "
    "Me gustaria que me ayudaran a aclararlo y, si es posible, bloquear la tarjeta "
    "mientras se resuelve. Tambien quisiera saber si puedo solicitar la reposicion "
    "sin costo y en cuanto tiempo me llegaria. Muchas gracias por su apoyo."
)
VOICES = ['es-MX-JorgeNeural', 'es-MX-DaliaNeural', 'es-ES-AlvaroNeural']


async def _gen(text, voice, out):
    comm = edge_tts.Communicate(text, voice)
    await comm.save(out)


def main():
    ap = argparse.ArgumentParser(description='Genera un audio sintetico (Edge-TTS es)')
    ap.add_argument('--text', default=DEFAULT_TEXT)
    ap.add_argument('--voice', default='es-MX-JorgeNeural', choices=VOICES + ['otra'])
    ap.add_argument('--voice-name', default=None, help='nombre exacto si --voice otra')
    ap.add_argument('--out', default='sintetico.mp3')
    args = ap.parse_args()
    voice = args.voice_name if args.voice == 'otra' and args.voice_name else args.voice
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_gen(args.text, voice, args.out))
    print(f'Generado: {args.out}  (voz {voice})')
    print(f'Analizalo:  .venv\\Scripts\\python.exe scripts/analyze_audio.py {args.out}')


if __name__ == '__main__':
    main()
