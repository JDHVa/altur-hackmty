import os
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get('ALTUR_DISPLAY_PORT', '8080'))
IDLE_AFTER = float(os.environ.get('ALTUR_IDLE_AFTER', '8'))

HW = None
tft = None
matrix = None
S = {'idle': 0, 'listening': 1, 'human': 2, 'bot': 3}
_last = time.time()
_lock = threading.Lock()

try:
    import centinela as HW
    tft = HW.init_tft()
    matrix = HW.init_matrix()
    print('Hardware Centinela OK (TFT + matriz)')
except Exception as e:
    print('Sin hardware fisico (modo consola):', str(e)[:100])


def paint(state, confidence, label):
    st = S.get(state, 0)
    if HW and tft is not None:
        try:
            HW.render_tft(tft, st, confidence, label)
            HW.render_matrix_icon(matrix, st)
        except Exception as e:
            print('render error:', str(e)[:80])
    else:
        bar = '#' * int(26 * max(0.0, min(1.0, confidence)))
        print(f'[{state.upper():<10}] {bar:<26} {confidence * 100:5.1f}% {label}', flush=True)


def watchdog():
    global _last
    while True:
        time.sleep(1)
        with _lock:
            idle = time.time() - _last > IDLE_AFTER
        if idle:
            paint('idle', 0.0, 'LISTO')
            with _lock:
                _last = time.time() + 1e9


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == '/health':
            self._send(200, {'status': 'ok', 'hardware': bool(HW)})
        else:
            self._send(404, {'error': 'not found'})

    def do_POST(self):
        global _last
        if self.path != '/state':
            self._send(404, {'error': 'not found'}); return
        try:
            n = int(self.headers.get('Content-Length', 0))
            d = json.loads(self.rfile.read(n))
        except Exception:
            self._send(400, {'error': 'bad json'}); return
        state = d.get('state', 'idle')
        conf = float(d.get('confidence', 0.0))
        label = d.get('label', '')
        with _lock:
            _last = time.time()
        paint(state, conf, label)
        self._send(200, {'ok': True})


def main():
    paint('idle', 0.0, 'LISTO')
    threading.Thread(target=watchdog, daemon=True).start()
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), H)
    print(f'Receptor Centinela escuchando en 0.0.0.0:{PORT}  (POST /state)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if HW and matrix is not None:
            try:
                matrix.clear()
            except Exception:
                pass


if __name__ == '__main__':
    main()
