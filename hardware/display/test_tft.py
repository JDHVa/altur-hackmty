from luma.core.interface.serial import spi
from luma.lcd.device import ili9341
from PIL import Image, ImageDraw

serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25)
tft = ili9341(serial, width=320, height=240, rotate=0)

img = Image.new("RGB", (320, 240), color=(10, 25, 47))
draw = ImageDraw.Draw(img)
draw.rectangle([(10, 10), (310, 230)], outline=(0, 220, 130), width=3)
draw.text((60, 70), "CENTINELA ALTUR", fill=(255, 255, 255))
draw.text((80, 110), "TEST DE PANTALLA", fill=(0, 220, 130))
draw.text((95, 150), "RPi 5 ONLINE", fill=(100, 180, 255))

tft.display(img)
