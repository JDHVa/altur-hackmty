import threading
import queue
import time
import sys
import signal
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas as matrix_canvas
from luma.lcd.device import ili9341

STATE_IDLE = 0
STATE_LISTENING = 1
STATE_RESULT_HUMAN = 2
STATE_RESULT_BOT = 3

TFT_W = 240
TFT_H = 320
SAFE_Y = 320
BUS_SPEED = 8000000

AUDIO_SR = 16000
ATTENUATION = 0.20
AUDIO_OUT = "plughw:2,0"
AUDIO_IN = "plughw:3,0"

COLOR_BG = (10, 15, 30)
COLOR_GREEN = (0, 220, 130)
COLOR_RED = (255, 50, 50)
COLOR_YELLOW = (255, 200, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (80, 80, 100)
COLOR_DARK = (30, 30, 50)
COLOR_FRAME = (50, 50, 80)

spi_lock = threading.Lock()
audio_queue = queue.Queue()
running = threading.Event()
running.set()

try:
    font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    font_lg = ImageFont.load_default()
    font_md = font_lg
    font_sm = font_lg


def init_tft():
    serial = spi(port=0, device=0, gpio_DC=24, gpio_RST=25, bus_speed_hz=BUS_SPEED)
    return ili9341(serial, width=TFT_W, height=TFT_H, rotate=0)


def init_matrix():
    serial = spi(port=0, device=1, gpio=noop())
    device = max7219(serial, cascaded=1, block_orientation=0, rotate=0)
    device.contrast(30)
    return device


def render_tft(device, state, confidence=0.0, label="IDLE"):
    safe = Image.new("RGB", (TFT_W, SAFE_Y), color=COLOR_BG)
    draw = ImageDraw.Draw(safe)

    header_text = "R3V0LUT10N"
    bbox = draw.textbbox((0, 0), header_text, font=font_lg)
    header_w = bbox[2] - bbox[0]
    draw.text(((TFT_W - header_w) // 2, 4), header_text, fill=COLOR_GREEN, font=font_lg)
    draw.line([(10, 28), (TFT_W - 10, 28)], fill=COLOR_GREEN, width=1)

    if state == STATE_IDLE:
        status_text = "SISTEMA LISTO"
        status_color = COLOR_GRAY
    elif state == STATE_LISTENING:
        status_text = "ANALIZANDO AUDIO..."
        status_color = COLOR_YELLOW
    elif state == STATE_RESULT_HUMAN:
        status_text = "HUMANO CONFIRMADO"
        status_color = COLOR_GREEN
    else:
        status_text = "VOZ SINTETICA / BOT"
        status_color = COLOR_RED

    draw.text((15, 34), status_text, fill=status_color, font=font_md)

    draw.rectangle([(10, 55), (TFT_W - 10, 115)], outline=COLOR_FRAME, width=1)

    draw.rectangle([(15, 65), (TFT_W - 15, 88)], fill=COLOR_DARK)

    clamped = min(max(confidence, 0.0), 1.0)
    bar_w = int((TFT_W - 30) * clamped)
    if bar_w > 0:
        if state == STATE_RESULT_HUMAN:
            bar_color = COLOR_GREEN
        elif state == STATE_RESULT_BOT:
            bar_color = COLOR_RED
        else:
            bar_color = COLOR_YELLOW
        draw.rectangle([(15, 65), (15 + bar_w, 88)], fill=bar_color)

    pct_str = f"{label} — {clamped * 100:.1f}%"
    draw.text((15, 93), pct_str, fill=COLOR_WHITE, font=font_md)

    draw.text((15, 130), time.strftime("%H:%M:%S"), fill=COLOR_GRAY, font=font_sm)
    draw.text((TFT_W - 110, 130), "CENTINELA", fill=COLOR_FRAME, font=font_sm)

    full = Image.new("RGB", (TFT_W, TFT_H), color=(0, 0, 0))
    full.paste(safe, (0, 0))

    with spi_lock:
        device.display(full)


def render_matrix_icon(device, state):
    with spi_lock:
        if state == STATE_IDLE:
            device.clear()
            return

        with matrix_canvas(device) as draw:
            if state == STATE_LISTENING:
                draw.line((1, 0, 4, 0), fill="white")
                draw.point((0, 1), fill="white")
                draw.point((5, 1), fill="white")
                draw.point((5, 2), fill="white")
                draw.point((4, 3), fill="white")
                draw.point((3, 4), fill="white")
                draw.point((3, 5), fill="white")
                draw.point((3, 7), fill="white")

            elif state == STATE_RESULT_HUMAN:
                draw.line((1, 4, 3, 6), fill="white")
                draw.line((3, 6, 6, 1), fill="white")

            elif state == STATE_RESULT_BOT:
                draw.line((1, 1, 6, 6), fill="white")
                draw.line((1, 6, 6, 1), fill="white")


def play_tone(freq, duration, vol=ATTENUATION):
    try:
        import sounddevice as sd
        t = np.linspace(0, duration, int(AUDIO_SR * duration), False)
        tone = (vol * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(tone, samplerate=AUDIO_SR, device=AUDIO_OUT)
        sd.wait()
    except Exception:
        pass


def play_sequence(freqs_durations, vol=ATTENUATION):
    for freq, dur in freqs_durations:
        play_tone(freq, dur, vol)
        time.sleep(0.05)


def record_audio(duration):
    try:
        import sounddevice as sd
        audio = sd.rec(
            int(duration * AUDIO_SR),
            samplerate=AUDIO_SR,
            channels=1,
            dtype="float32",
            device=AUDIO_IN,
        )
        sd.wait()
        return audio.flatten()
    except Exception:
        return np.zeros(int(duration * AUDIO_SR), dtype=np.float32)


def playback_audio(audio, vol=ATTENUATION):
    try:
        import sounddevice as sd
        attenuated = (audio * vol).astype(np.float32)
        sd.play(attenuated, samplerate=AUDIO_SR, device=AUDIO_OUT)
        sd.wait()
    except Exception:
        pass


def audio_worker():
    while running.is_set():
        try:
            task = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if task is None:
            break
        action = task.get("action")
        if action == "tone":
            play_tone(task["freq"], task["duration"], task.get("vol", ATTENUATION))
        elif action == "sequence":
            play_sequence(task["freqs"], task.get("vol", ATTENUATION))
        elif action == "record":
            audio = record_audio(task["duration"])
            if "callback" in task:
                task["callback"](audio)
        elif action == "play":
            playback_audio(task["audio"], task.get("vol", ATTENUATION))
        audio_queue.task_done()


def demo_cycle(tft, matrix):
    render_tft(tft, STATE_IDLE, 0.0, "IDLE")
    render_matrix_icon(matrix, STATE_IDLE)
    time.sleep(2)

    if not running.is_set():
        return

    render_tft(tft, STATE_LISTENING, 0.0, "LISTENING")
    render_matrix_icon(matrix, STATE_LISTENING)
    play_tone(880, 0.15)
    time.sleep(0.5)

    for i in range(25):
        if not running.is_set():
            return
        progress = i / 25.0
        render_tft(tft, STATE_LISTENING, progress, "ANALYZING")
        time.sleep(0.12)

    if not running.is_set():
        return

    is_human = random.random() > 0.5
    final_conf = random.uniform(0.72, 0.98)

    if is_human:
        render_tft(tft, STATE_RESULT_HUMAN, final_conf, "HUMAN")
        render_matrix_icon(matrix, STATE_RESULT_HUMAN)
        play_sequence([(523, 0.15), (659, 0.15), (784, 0.25)])
    else:
        render_tft(tft, STATE_RESULT_BOT, final_conf, "BOT")
        render_matrix_icon(matrix, STATE_RESULT_BOT)
        play_sequence([(200, 0.4), (150, 0.4)])

    time.sleep(4)


def demo_loop(tft, matrix):
    while running.is_set():
        demo_cycle(tft, matrix)


def live_loop(tft, matrix):
    render_tft(tft, STATE_IDLE, 0.0, "READY")
    render_matrix_icon(matrix, STATE_IDLE)
    time.sleep(1)

    while running.is_set():
        render_tft(tft, STATE_LISTENING, 0.0, "RECORDING")
        render_matrix_icon(matrix, STATE_LISTENING)
        play_tone(880, 0.1)

        audio = record_audio(5.0)

        if not running.is_set():
            break

        rms = float(np.sqrt(np.mean(audio ** 2)))
        fake_confidence = min(rms * 8.0, 1.0)

        for i in range(20):
            if not running.is_set():
                return
            progress = i / 20.0
            render_tft(tft, STATE_LISTENING, progress, "ANALYZING")
            time.sleep(0.08)

        is_human = fake_confidence < 0.5
        if is_human:
            render_tft(tft, STATE_RESULT_HUMAN, 1.0 - fake_confidence, "HUMAN")
            render_matrix_icon(matrix, STATE_RESULT_HUMAN)
            play_sequence([(523, 0.15), (784, 0.2)])
        else:
            render_tft(tft, STATE_RESULT_BOT, fake_confidence, "BOT")
            render_matrix_icon(matrix, STATE_RESULT_BOT)
            play_sequence([(200, 0.3), (150, 0.3)])

        time.sleep(3)


def main():
    def signal_handler(sig, frame):
        running.clear()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    tft = init_tft()
    matrix = init_matrix()

    mode = "demo"
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    try:
        if mode == "live":
            live_loop(tft, matrix)
        else:
            demo_loop(tft, matrix)
    finally:
        with spi_lock:
            matrix.clear()
        render_tft(tft, STATE_IDLE, 0.0, "OFFLINE")


if __name__ == "__main__":
    main()
