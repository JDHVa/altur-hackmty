import time
from luma.core.interface.serial import spi
from luma.lcd.device import ili9341
from PIL import Image, ImageDraw

serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25)
tft = ili9341(serial, width=240, height=320, rotate=0)

img = Image.new("RGB", (240, 320), color=(10, 25, 47))
draw = ImageDraw.Draw(img)
draw.rectangle([(5, 5), (235, 315)], outline=(0, 255, 128), width=4)
draw.rectangle([(20, 20), (220, 70)], fill=(0, 100, 255))
draw.rectangle([(20, 80), (220, 130)], fill=(0, 200, 100))
draw.rectangle([(20, 140), (220, 190)], fill=(255, 50, 50))
draw.text((30, 35), "CENTINELA ALTUR", fill=(255, 255, 255))
draw.text((30, 95), "HARDWARE ACTIVO", fill=(0, 0, 0))
draw.text((30, 155), "ESTADO: OK", fill=(255, 255, 255))

tft.display(img)
time.sleep(10)
