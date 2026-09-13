import time
from luma.core.interface.serial import spi
from luma.lcd.device import ili9341
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', sz)
    except Exception:
        return ImageFont.load_default()


F_BIG = font(40)
F_MD = font(22)


def draw(rot):
    serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25, bus_speed_hz=8000000)
    dev = ili9341(serial, width=320, height=240, rotate=rot)
    W, H = dev.width, dev.height
    img = Image.new('RGB', (W, H), (10, 12, 22))
    d = ImageDraw.Draw(img)
    d.rectangle([(3, 3), (W - 4, H - 4)], outline=(120, 200, 120), width=3)
    def ctr(txt, y, f, col):
        bb = d.textbbox((0, 0), txt, font=f)
        d.text(((W - (bb[2] - bb[0])) // 2, y), txt, font=f, fill=col)
    ctr('ARRIBA', 12, F_MD, (120, 220, 120))
    ctr(f'r = {rot}', H // 2 - 30, F_BIG, (230, 120, 240))
    ctr(f'{W} x {H}', H // 2 + 22, F_MD, (200, 200, 210))
    ctr('ABAJO', H - 40, F_MD, (200, 120, 120))
    with __import__('threading').Lock():
        dev.display(img)
    return dev


def main():
    print('Mostrando r=0,1,2,3 (5s cada uno). Anota cual sale PARADA y con ARRIBA arriba.')
    for r in [0, 1, 2, 3]:
        print(f'  rotate = {r}')
        draw(r)
        time.sleep(5)
    print('Listo. Corre:  ALTUR_TFT_ROTATE=<N> python3 centinela_display.py')


if __name__ == '__main__':
    main()
