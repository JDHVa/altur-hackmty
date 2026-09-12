import time
import numpy as np
import sounddevice as sd

sr = 16000
duration = 2.0
t = np.linspace(0, duration, int(sr * duration), False)
tone = 0.20 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

device_out = "plughw:2,0"
device_in = "plughw:3,0"

try:
    sd.play(tone, samplerate=sr, device=device_out)
    sd.wait()
except Exception:
    sd.play(tone, samplerate=sr)
    sd.wait()

time.sleep(0.5)

try:
    rec = sd.rec(int(3.0 * sr), samplerate=sr, channels=1, dtype="float32", device=device_in)
    sd.wait()
    rec_scaled = rec * 0.4
    sd.play(rec_scaled, samplerate=sr, device=device_out)
    sd.wait()
except Exception:
    rec = sd.rec(int(3.0 * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    rec_scaled = rec * 0.4
    sd.play(rec_scaled, samplerate=sr)
    sd.wait()
