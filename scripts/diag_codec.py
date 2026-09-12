# -*- coding: utf-8 -*-
import os
import sys
import glob
import subprocess
import numpy as np
import soundfile as sf
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.audio import audio_score
from features.telephony_aug import apply_codec, CODECS

SRC_DIR = r"C:\Users\emili\OneDrive\Documentos\Grabaciones de sonido"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_m4a(path):
    dst = os.path.join(os.path.dirname(__file__), '..', '_tmp_wav',
                       os.path.splitext(os.path.basename(path))[0] + '_orig16k.wav')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run([FFMPEG, '-y', '-i', path, '-ac', '1', '-ar', '16000', dst],
                   check=True, capture_output=True)
    w, sr = sf.read(dst)
    return (w[:, 0] if w.ndim > 1 else w).astype(np.float32), sr


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, '*.m4a')))
    print(f"Diagnostico codec sobre {len(files)} grabaciones humanas propias\n")
    print(f"{'archivo':16s} {'limpio16k':>10s} " + ' '.join(f'{c:>10s}' for c in CODECS))
    for f in files:
        c16, sr = load_m4a(f)
        base = audio_score(c16, sr)
        row = [f'{base:.3f}']
        for codec in CODECS:
            cc, ccsr = apply_codec(c16, sr, codec)
            row.append(f'{audio_score(cc, ccsr):.3f}')
        name = os.path.basename(f)[:16]
        print(f"{name:16s} " + ' '.join(f'{v:>10s}' for v in row))
    print("\n(score = prob sintetico; estas 3 son HUMANAS -> lo ideal es < 0.5)")


if __name__ == '__main__':
    main()
