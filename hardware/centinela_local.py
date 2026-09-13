import os
import io
import sys
import json
import time
import wave
import base64
import urllib.request
import numpy as np

LOCAL_URL = os.environ.get('ALTUR_LOCAL_URL', 'http://127.0.0.1:8010')
PI_URL = os.environ.get('ALTUR_PI_URL', '')
MIC_DEVICE = os.environ.get('ALTUR_MIC_DEVICE', '')
START_THR = float(os.environ.get('ALTUR_VAD_THR', '0.018'))
SILENCE_S = float(os.environ.get('ALTUR_VAD_SILENCE', '0.8'))
MIN_SPEECH_S = float(os.environ.get('ALTUR_VAD_MIN', '0.8'))
MAX_SEG_S = float(os.environ.get('ALTUR_VAD_MAX', '15'))
DIRECT = '--direct' in sys.argv

_audio_only = None


def load_direct():
    global _audio_only
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', 'altur_100'))
    sys.path.insert(0, os.path.join(here, '..', 'altur_100', 'src'))
    from pipeline import audio_only
    _audio_only = audio_only


def wav_b64(audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()
    b = io.BytesIO()
    w = wave.open(b, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm); w.close()
    return base64.b64encode(b.getvalue()).decode()


def classify(audio, sr):
    if DIRECT:
        d = _audio_only(audio.reshape(-1, 1), sr)
        return float(d['p_synthetic'])
    body = json.dumps({'audio_base64': wav_b64(audio, sr)}).encode()
    req = urllib.request.Request(LOCAL_URL + '/detect/audio', data=body, headers={'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return float(d['p_synthetic'])


def push_pi(state, confidence=0.0, label='—'):
    if not PI_URL:
        return
    try:
        body = json.dumps({'state': state, 'confidence': confidence, 'label': label}).encode()
        req = urllib.request.Request(PI_URL, data=body, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=1.5).read()
    except Exception:
        pass


BAR_W = 26


def show(state, confidence=0.0, label='—'):
    fill = int(BAR_W * max(0.0, min(1.0, confidence)))
    bar = '#' * fill + '-' * (BAR_W - fill)
    icon = {'idle': '  ', 'listening': '..', 'human': 'OK', 'bot': '!!'}.get(state, '  ')
    line = f'\r[{icon}] {state.upper():<10} [{bar}] {confidence * 100:5.1f}%  {label:<8}'
    sys.stdout.write(line + ' ' * 6)
    sys.stdout.flush()


def check_local():
    if DIRECT:
        return True
    try:
        urllib.request.urlopen(LOCAL_URL + '/health', timeout=5).read()
        return True
    except Exception:
        return False


def main():
    try:
        import sounddevice as sd
    except Exception:
        print('Falta sounddevice. Instala: pip install sounddevice')
        return

    if DIRECT:
        print('Cargando modelo local en proceso (--direct)...')
        try:
            load_direct()
        except Exception as e:
            print('No se pudo cargar el modelo local:', str(e)[:120]); return
    elif not check_local():
        print(f'Servidor local no responde en {LOCAL_URL}. Levantalo (uvicorn serve:app --port 8010) o usa --direct.')
        return

    dev = int(MIC_DEVICE) if MIC_DEVICE.isdigit() else (MIC_DEVICE or None)
    try:
        info = sd.query_devices(dev, 'input')
        sr = int(info['default_samplerate'])
    except Exception as e:
        print('No hay microfono disponible:', str(e)[:120]); return

    print(f'Centinela LOCAL activo. mic="{info["name"]}" sr={sr} destino={"in-proc" if DIRECT else LOCAL_URL} pi={PI_URL or "(consola)"}')
    print('Escuchando... (Ctrl+C para salir)')

    frame = int(sr * 0.1)
    speaking = False
    buf = []
    silence = 0.0
    seg = 0.0
    last_idle = 0.0
    push_pi('idle')

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32', device=dev, blocksize=frame) as stream:
            while True:
                block, _ = stream.read(frame)
                x = block[:, 0]
                rms = float(np.sqrt(np.mean(x * x)) + 1e-9)

                if not speaking:
                    if rms > START_THR:
                        speaking = True; buf = [x.copy()]; silence = 0.0; seg = 0.1
                        show('listening', 0.0, 'escuchando'); push_pi('listening')
                    else:
                        now = time.time()
                        if now - last_idle > 1.0:
                            show('idle', 0.0, 'listo'); last_idle = now
                    continue

                buf.append(x.copy()); seg += 0.1
                if rms < START_THR:
                    silence += 0.1
                else:
                    silence = 0.0

                if silence >= SILENCE_S or seg >= MAX_SEG_S:
                    speaking = False
                    audio = np.concatenate(buf).astype(np.float32)
                    buf = []
                    if len(audio) / sr < MIN_SPEECH_S:
                        show('idle', 0.0, 'listo'); push_pi('idle'); continue
                    show('listening', 0.0, 'analizando'); push_pi('listening')
                    try:
                        p = classify(audio, sr)
                    except Exception as e:
                        sys.stdout.write('\n  error clasificando: ' + str(e)[:100] + '\n'); continue
                    if p >= 0.5:
                        show('bot', p, 'IA'); push_pi('bot', p, 'VOZ IA')
                    else:
                        show('human', 1.0 - p, 'HUMANO'); push_pi('human', 1.0 - p, 'HUMANO')
                    time.sleep(1.0)
    except KeyboardInterrupt:
        push_pi('idle')
        print('\nSaliendo.')


if __name__ == '__main__':
    main()
