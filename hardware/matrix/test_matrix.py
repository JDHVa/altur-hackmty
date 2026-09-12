import time
from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas

serial = spi(port=0, device=1, gpio=noop())
matrix = max7219(serial, cascaded=1, block_orientation=0, rotate=0)
matrix.contrast(30)

with canvas(matrix) as draw:
    draw.rectangle(matrix.bounding_box, outline="white", fill="white")
time.sleep(2)

with canvas(matrix) as draw:
    draw.line((1, 4, 3, 6), fill="white")
    draw.line((3, 6, 6, 1), fill="white")
time.sleep(2)

with canvas(matrix) as draw:
    draw.line((1, 1, 6, 6), fill="white")
    draw.line((1, 6, 6, 1), fill="white")
time.sleep(2)

matrix.clear()
