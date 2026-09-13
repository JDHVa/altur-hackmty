import sys
import time
import numpy as np
import sounddevice as sd

sr = 16000
t = np.linspace(0, 1.0, sr, False)
tone = (0.4 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

print('=== dispositivos de audio ===')
print(sd.query_devices())
print('=============================')

cands = sys.argv[1:] if len(sys.argv) > 1 else ['plughw:2,0', 'default', None, 0, 1, 2, 3]
for dev in cands:
    d = None if dev in (None, 'None') else (int(dev) if str(dev).isdigit() else dev)
    print(f'\n>> probando salida: {dev!r}')
    try:
        sd.play(tone, samplerate=sr, device=d); sd.wait()
        print('   sono? (deberias haber oido 1s de tono)')
    except Exception as e:
        print('   FALLA:', type(e).__name__, str(e)[:120])
    time.sleep(0.4)

print('\nCuando sepas cual sono, corre el receptor asi:')
print('  ALTUR_AUDIO_OUT="<ese_device>" python3 centinela_display.py')
