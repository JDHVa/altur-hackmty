# -*- coding: utf-8 -*-
import os
import sys
import glob
import subprocess
import numpy as np
import soundfile as sf
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features.audio import audio_score, audio_features

SRC_DIR = r"C:\Users\emili\OneDrive\Documentos\Grabaciones de sonido"
TMP = os.environ.get("TMP_WAV_DIR", os.path.join(os.path.dirname(__file__), '..', '_tmp_wav'))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def to_wav(src):
    os.makedirs(TMP, exist_ok=True)
    dst = os.path.join(TMP, os.path.splitext(os.path.basename(src))[0] + ".wav")
    subprocess.run([FFMPEG, "-y", "-i", src, "-ac", "1", "-ar", "16000", dst],
                   check=True, capture_output=True)
    return dst


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.m4a")))
    if not files:
        print("No se encontraron .m4a en", SRC_DIR)
        return
    print(f"Modelo: ResNet audio (canal 0 = voz). {len(files)} archivos.\n")
    for f in files:
        wav = to_wav(f)
        w, sr = sf.read(wav)
        caller = (w[:, 0] if w.ndim > 1 else w).astype(np.float32)
        dur = len(caller) / sr
        score = audio_score(caller, sr)
        feats = audio_features(caller, sr)
        verdict = "SINTETICO (IA)" if score >= 0.5 else "HUMANO"
        print("=" * 60)
        print(f"{os.path.basename(f)}  ({dur:.1f}s)")
        print(f"  score_sintetico = {score:.3f}  ->  {verdict}")
        print(f"  f0_mean={feats['f0_mean']:.0f}Hz  f0_std={feats['f0_std']:.1f}  "
              f"voiced={feats['voiced_ratio']:.2f}  rms_std={feats['rms_std']:.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
