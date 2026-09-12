import io
import os
import sys
import json
import subprocess
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pipeline import predict


def read_audio(path):
    try:
        return sf.read(path, always_2d=True, dtype='float32')
    except Exception:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        p = subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-i', path, '-f', 'wav', 'pipe:1'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return sf.read(io.BytesIO(p.stdout), always_2d=True, dtype='float32')


def predict_file(path):
    data, sr = read_audio(path)
    return predict(data, sr)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('uso: python infer.py <archivo.wav>')
        sys.exit(1)
    print(json.dumps(predict_file(sys.argv[1]), ensure_ascii=False, indent=2))
