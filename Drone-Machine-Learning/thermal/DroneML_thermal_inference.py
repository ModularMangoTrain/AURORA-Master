import sys
import os
import signal
import time
import logging
import numpy as np
import cv2 as cv

from smbus2 import SMBus
from spidev import SpiDev
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

from senxor.mi48 import MI48, DATA_READY
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))

RPI_GPIO_I2C_CHANNEL  = 1
RPI_GPIO_SPI_BUS      = 0
RPI_GPIO_SPI_CE_MI48  = 1
MI48_I2C_ADDRESS      = 0x40
MI48_SPI_MODE         = 0b00
MI48_SPI_MAX_SPEED_HZ = 31200000

# Blob detection tuning - adjust these if getting false positives/negatives
HOT_PERCENTILE   = 85   # pixels above this percentile are considered "hot"
MIN_BLOB_PIXELS  = 8    # minimum blob area in raw 80x62 pixels
MAX_BLOB_PIXELS  = 2000 # maximum blob area
MIN_ASPECT       = 0.3  # min height/width ratio
MAX_ASPECT       = 6.0  # max height/width ratio

def detect_persons(img8u):
    """Detect hot blobs in 80x62 thermal frame. Returns list of (x1,y1,x2,y2) in raw coords."""
    threshold = np.percentile(img8u, HOT_PERCENTILE)
    mask = (img8u >= threshold).astype(np.uint8) * 255
    # Morphological close to join nearby hot pixels
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_BLOB_PIXELS or area > MAX_BLOB_PIXELS:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        aspect = h / max(w, 1)
        if MIN_ASPECT <= aspect <= MAX_ASPECT:
            detections.append((x, y, x + w, y + h))
    return detections

# Setup MI48
factory = LGPIOFactory(chip=4)
i2c = I2C_Interface(SMBus(RPI_GPIO_I2C_CHANNEL), MI48_I2C_ADDRESS)
spi = SPI_Interface(SpiDev(RPI_GPIO_SPI_BUS, RPI_GPIO_SPI_CE_MI48), xfer_size=160)
spi.device.mode          = MI48_SPI_MODE
spi.device.max_speed_hz  = MI48_SPI_MAX_SPEED_HZ
spi.device.bits_per_word = 8
spi.device.lsbfirst      = False

mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False, pin_factory=factory)
mi48_reset_n    = DigitalOutputDevice("BCM23", active_high=False, initial_value=True, pin_factory=factory)

class MI48_reset:
    def __init__(self, pin, assert_seconds=0.000035, deassert_seconds=0.050):
        self.pin = pin
        self.assert_time = assert_seconds
        self.deassert_time = deassert_seconds
    def __call__(self):
        self.pin.on()
        time.sleep(self.assert_time)
        self.pin.off()
        time.sleep(self.deassert_time)

mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
            reset_handler=MI48_reset(pin=mi48_reset_n))
mi48.set_fps(9)
if int(mi48.fw_version[0]) >= 2:
    mi48.enable_filter(f1=True, f2=True, f3=False)
    mi48.set_offset_corr(0.0)

def signal_handler(sig, frame):
    mi48.stop(poll_timeout=0.25, stop_timeout=1.2)
    cv.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

mi48.start(stream=True, with_header=True)
print("Thermal SAR - blob detection mode. Press 'q' to quit, 's' to save")

DISPLAY_SIZE = (640, 512)
scale_x = DISPLAY_SIZE[0] / 80
scale_y = DISPLAY_SIZE[1] / 62
frame_count = 0
WARMUP_FRAMES = 10

while True:
    if hasattr(mi48, 'data_ready'):
        mi48.data_ready.wait_for_active()
    else:
        data_ready = False
        while not data_ready:
            time.sleep(0.01)
            data_ready = mi48.get_status() & DATA_READY

    data, header = mi48.read()
    img = data_to_frame(data, mi48.fpa_shape)

    img_float = img.astype(np.float32)
    p2, p98 = np.percentile(img_float, 2), np.percentile(img_float, 98)
    img8u = np.clip((img_float - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    img8u = cv_filter(img8u, parameters={'blur_ks': 3}, use_median=False, use_bilat=True, use_nlm=False)

    frame_count += 1
    if frame_count < WARMUP_FRAMES:
        continue

    detections = detect_persons(img8u)

    # Display
    display = cv.applyColorMap(img8u, cv.COLORMAP_JET)
    display = cv.resize(display, DISPLAY_SIZE, interpolation=cv.INTER_CUBIC)

    for (x1, y1, x2, y2) in detections:
        dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
        dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
        cv.rectangle(display, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
        cv.putText(display, "Person", (dx1, dy1 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    person_count = len(detections)
    status = f"SURVIVORS: {person_count}" if person_count > 0 else "SEARCHING..."
    color  = (0, 255, 0) if person_count > 0 else (0, 0, 255)
    cv.putText(display, status, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.putText(display, "THERMAL MODE", (10, DISPLAY_SIZE[1] - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    cv.imshow('Thermal Search and Rescue', display)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"thermal_rescue_{frame_count}.jpg"
        cv.imwrite(filename, display)
        print(f"Saved {filename} - {person_count} person(s) detected")

mi48.stop(stop_timeout=0.5)
cv.destroyAllWindows()
