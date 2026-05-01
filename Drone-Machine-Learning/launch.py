#!/usr/bin/env python3
import time
import signal
import sys
import threading
import subprocess
import spidev

NUM_LEDS = 144

SUNRISE_SUNSET = {
    1:  (8, 0,  16, 2),
    2:  (7, 30, 17, 0),
    3:  (6, 30, 18, 0),
    4:  (6, 15, 20, 0),
    5:  (5, 30, 20, 45),
    6:  (4, 45, 21, 20),
    7:  (5, 0,  21, 10),
    8:  (5, 45, 20, 20),
    9:  (6, 30, 19, 10),
    10: (7, 15, 18, 0),
    11: (7, 0,  16, 10),
    12: (8, 0,  15, 55),
}

def is_daytime():
    now = time.localtime()
    sr_h, sr_m, ss_h, ss_m = SUNRISE_SUNSET[now.tm_mon]
    current = now.tm_hour * 60 + now.tm_min
    return (sr_h * 60 + sr_m) <= current <= (ss_h * 60 + ss_m)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 3200000
spi.mode = 0

def encode_byte(byte):
    result = []
    for i in range(7, -1, -1):
        result.append(0b11100000 if byte & (1 << i) else 0b10000000)
    return result

def encode_rgb(r, g, b):
    return encode_byte(g) + encode_byte(r) + encode_byte(b)

def show(pixels):
    data = []
    for r, g, b in pixels:
        data += encode_rgb(r, g, b)
    data += [0] * 10
    spi.xfer2(data)

# Shared state
survivor_detected = threading.Event()
FLASH_DURATION = 10  # seconds to flash after detection

def led_thread():
    flash_until = 0
    while True:
        if survivor_detected.is_set():
            flash_until = time.time() + FLASH_DURATION
            survivor_detected.clear()

        if time.time() < flash_until:
            show([(255, 255, 255)] * NUM_LEDS)  # white flash
            time.sleep(0.15)
            show([(0, 0, 0)] * NUM_LEDS)
            time.sleep(0.15)
        else:
            color = (255, 220, 0) if is_daytime() else (255, 0, 0)
            show([color] * NUM_LEDS)
            time.sleep(1)

def sigint_handler(sig, frame):
    show([(0, 0, 0)] * NUM_LEDS)
    spi.close()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

threading.Thread(target=led_thread, daemon=True).start()

# Launch CNN.py and watch stdout for survivor alert
proc = subprocess.Popen(
    ['python3','-u', 'CNN.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Launched CNN.py - watching for survivor alerts...")

for line in proc.stdout:
    print(line, end='')  # pass through all output
    if 'CONFIRMED SURVIVOR' in line:
        print("[LAUNCH] Survivor detected - triggering LED flash!")
        survivor_detected.set()

proc.wait()
show([(0, 0, 0)] * NUM_LEDS)
spi.close()
