#!/usr/bin/env python3
import time
import signal
import sys
import spidev

NUM_LEDS = 144
TAIL = 12

# BST sunrise/sunset by month (24hr)
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
    month = now.tm_mon
    hour, minute = now.tm_hour, now.tm_min
    sr_h, sr_m, ss_h, ss_m = SUNRISE_SUNSET[month]
    current = hour * 60 + minute
    sunrise = sr_h * 60 + sr_m
    sunset  = ss_h * 60 + ss_m
    return sunrise <= current <= sunset

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 3200000
spi.mode = 0

def encode_byte(byte):
    result = []
    for i in range(7, -1, -1):
        if byte & (1 << i):
            result.append(0b11100000)  # 1 bit: 937ns high, 312ns low
        else:
            result.append(0b10000000)  # 0 bit: 312ns high, 937ns low
    return result

def encode_rgb(r, g, b):
    return encode_byte(g) + encode_byte(r) + encode_byte(b)

def show(pixels):
    data = []
    for r, g, b in pixels:
        data += encode_rgb(r, g, b)
    data += [0] * 10  # reset
    spi.xfer2(data)

def sigint_handler(sig, frame):
    show([(0, 0, 0)] * NUM_LEDS)
    spi.close()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

while True:
    color = (255, 220, 0) if is_daytime() else (255, 0, 0)  # neon yellow : bright red
    show([color] * NUM_LEDS)
    time.sleep(1)
