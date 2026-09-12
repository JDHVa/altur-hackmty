# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.features.telephony_aug import augment_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--config', default='es_419')
    ap.add_argument('--split', default='train')
    ap.add_argument('--codec', default='g711_ulaw')
    ap.add_argument('--out', default='datasets_externos/Human_ES_8kHz')
    args = ap.parse_args()

    import io
    from datasets import load_dataset, Audio
    os.makedirs(args.out, exist_ok=True)
    print(f'Streaming FLEURS {args.config}/{args.split} -> {args.n} humanos, canal {args.codec}')
    ds = load_dataset('google/fleurs', args.config, split=args.split, streaming=True)
    ds = ds.cast_column('audio', Audio(decode=False))

    saved = 0
    for ex in ds:
        if saved >= args.n:
            break
        audio = ex['audio']
        data = audio['bytes'] if audio.get('bytes') else open(audio['path'], 'rb').read()
        wave, sr = sf.read(io.BytesIO(data))
        wave = (wave[:, 0] if wave.ndim > 1 else wave).astype(np.float32)
        if wave.size < sr:
            continue
        proc, out_sr = augment_chain(wave, sr, codec=args.codec)
        path = os.path.join(args.out, f'human_fleurs_{saved:03d}.wav')
        sf.write(path, proc, out_sr, subtype='PCM_16')
        saved += 1
        if saved % 10 == 0:
            print(f'  [{saved}/{args.n}]', flush=True)
    print(f'Listo. {saved} humanos es en {args.out}')


if __name__ == '__main__':
    main()
