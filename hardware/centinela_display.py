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

TFT_ROTATE = int(os.environ.get('ALTUR_TFT_ROTATE', '1'))
MATRIX_ROTATE = int(os.environ.get('ALTUR_MATRIX_ROTATE', '2'))

try:
    import centinela as HW
    from luma.core.interface.serial import spi, noop
    from luma.lcd.device import ili9341
    from luma.led_matrix.device import max7219
    serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25, bus_speed_hz=HW.BUS_SPEED)
    tft = ili9341(serial, width=320, height=240, rotate=TFT_ROTATE)
    mserial = spi(port=0, device=1, gpio=noop())
    matrix = max7219(mserial, cascaded=1, block_orientation=0, rotate=MATRIX_ROTATE)
    matrix.contrast(30)
    print(f'Hardware OK (TFT {tft.width}x{tft.height} rotate={TFT_ROTATE}, matriz rotate={MATRIX_ROTATE})')
except Exception as e:
    print('Sin hardware fisico (modo consola):', str(e)[:100])

COL = {'human': (150, 80, 230), 'bot': (235, 60, 60), 'listening': (240, 180, 40), 'idle': (120, 120, 140)}
TITLE = {'human': ['HUMANO'], 'bot': ['INTELIGENCIA', 'ARTIFICIAL'], 'listening': ['ANALIZANDO...'], 'idle': ['EN ESPERA']}
_F = {}

if HW:
    from PIL import Image, ImageDraw, ImageFont

    def _font(sz, bold=True):
        try:
            p = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
            return ImageFont.truetype(p, sz)
        except Exception:
            return ImageFont.load_default()
    _F = {'title': _font(26), 'sub': _font(17, False), 'small': _font(13, False)}


def render_tft_custom(state, conf, label):
    W, H = tft.width, tft.height
    img = Image.new('RGB', (W, H), (12, 12, 20))
    d = ImageDraw.Draw(img)
    d.text((10, 12), 'CENTINELA ALTUR', font=_F['small'], fill=(95, 95, 125))
    d.line([(10, 34), (W - 10, 34)], fill=(40, 40, 60), width=1)
    col = COL.get(state, (120, 120, 140))
    lines = TITLE.get(state, ['EN ESPERA'])
    y = int(H * 0.34) - len(lines) * 20
    for ln in lines:
        bb = d.textbbox((0, 0), ln, font=_F['title'])
        d.text(((W - (bb[2] - bb[0])) // 2, y), ln, font=_F['title'], fill=col)
        y += 40
    if state in ('human', 'bot'):
        by = y + 24
        d.rectangle([(20, by), (W - 20, by + 22)], fill=(30, 30, 45))
        bw = int((W - 44) * max(0.0, min(1.0, conf)))
        d.rectangle([(22, by + 2), (22 + bw, by + 20)], fill=col)
        pct = f'{conf * 100:.0f}%'
        bb = d.textbbox((0, 0), pct, font=_F['sub'])
        d.text(((W - (bb[2] - bb[0])) // 2, by + 30), pct, font=_F['sub'], fill=(230, 230, 235))
    with HW.spi_lock:
        tft.display(img)


def paint(state, confidence, label):
    if HW and tft is not None:
        try:
            render_tft_custom(state, confidence, label)
            if matrix is not None:
                HW.render_matrix_icon(matrix, S.get(state, 0))
        except Exception as e:
            print('render error:', str(e)[:80], flush=True)
    else:
        name = ' '.join(TITLE.get(state, [state.upper()]))
        bar = '#' * int(26 * max(0.0, min(1.0, confidence)))
        print(f'[{name:<24}] {bar:<26} {confidence * 100:5.1f}%', flush=True)


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
