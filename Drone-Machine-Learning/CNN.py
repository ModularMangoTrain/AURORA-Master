import sys
import os
import signal
import time
import threading
import multiprocessing
import logging
import numpy as np
import cv2 as cv
from queue import Queue
import spidev

from ultralytics import YOLO
from picamera2 import Picamera2
from smbus2 import SMBus
from spidev import SpiDev
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

from senxor.mi48 import MI48, DATA_READY
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/aurora/aurora_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- LED setup ---
NUM_LEDS = 144
SUNRISE_SUNSET = {
    1:  (8, 0,  16, 2),  2:  (7, 30, 17, 0),  3:  (6, 30, 18, 0),
    4:  (6, 15, 20, 0),  5:  (5, 30, 20, 45),  6:  (4, 45, 21, 20),
    7:  (5, 0,  21, 10), 8:  (5, 45, 20, 20),  9:  (6, 30, 19, 10),
    10: (7, 15, 18, 0),  11: (7, 0,  16, 10),  12: (8, 0,  15, 55),
}

def is_daytime():
    now = time.localtime()
    sr_h, sr_m, ss_h, ss_m = SUNRISE_SUNSET[now.tm_mon]
    current = now.tm_hour * 60 + now.tm_min
    return (sr_h * 60 + sr_m) <= current <= (ss_h * 60 + ss_m)

_led_spi = spidev.SpiDev()
_led_spi.open(0, 0)
_led_spi.max_speed_hz = 3200000
_led_spi.mode = 0

def _encode_byte(byte):
    result = []
    for i in range(7, -1, -1):
        result.append(0b11100000 if byte & (1 << i) else 0b10000000)
    return result

def led_show(pixels):
    data = []
    for r, g, b in pixels:
        data += _encode_byte(g) + _encode_byte(r) + _encode_byte(b)
    data += [0] * 10
    with _spi0_lock:
        _led_spi.xfer2(data)

FLASH_DURATION = 10
_survivor_event = threading.Event()
_led_lock = threading.Lock()
_spi0_lock = threading.Lock()

def _led_thread():
    flash_until = 0
    while True:
        try:
            if _survivor_event.is_set():
                flash_until = time.time() + FLASH_DURATION
                _survivor_event.clear()
                logger.info("LED: survivor flash triggered")
            if time.time() < flash_until:
                color = (10, 9, 0) if is_daytime() else (10, 0, 0)
                led_show([color] * NUM_LEDS)
                time.sleep(0.15)
                led_show([(0, 0, 0)] * NUM_LEDS)
                time.sleep(0.15)
            else:
                color = (255, 220, 0) if is_daytime() else (255, 0, 0)
                led_show([color] * NUM_LEDS)
                time.sleep(1)
        except Exception:
            logger.exception("LED thread crashed")
            time.sleep(1)

threading.Thread(target=_led_thread, daemon=True).start()
# --- end LED setup ---

# --- Fusion config ---
FUSION_WINDOW_SEC = 0.5

# --- Thermal blob config ---
HOT_PERCENTILE  = 85
MIN_BLOB_PIXELS = 8
MAX_BLOB_PIXELS = 2000
MIN_ASPECT      = 0.3
MAX_ASPECT      = 6.0
THERMAL_WARMUP  = 20

# --- Shared state ---
rgb_last_detection     = 0.0
thermal_last_detection = 0.0
state_lock = threading.Lock()
running = True
thermal_display_queue = Queue(maxsize=1)

# ─────────────────────────────────────────────
# Thermal blob detection
# ─────────────────────────────────────────────
def detect_blobs(img8u):
    threshold = np.percentile(img8u, HOT_PERCENTILE)
    mask = (img8u >= threshold).astype(np.uint8) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_BLOB_PIXELS or area > MAX_BLOB_PIXELS:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        if MIN_ASPECT <= (h / max(w, 1)) <= MAX_ASPECT:
            detections.append((x, y, x + w, y + h))
    return detections

# ─────────────────────────────────────────────
# Thermal stream thread
# ─────────────────────────────────────────────
def thermal_stream():
    global thermal_last_detection, running
    try:
        _run_thermal_stream()
    except Exception:
        logger.exception("Thermal thread crashed")

def _run_thermal_stream():
    global thermal_last_detection, running

    factory = LGPIOFactory(chip=4)
    i2c = I2C_Interface(SMBus(1), 0x40)
    _thermal_spi = SpiDev(0, 1)
    _thermal_spi.mode = 0b00
    _thermal_spi.max_speed_hz = 3900000
    _thermal_spi.bits_per_word = 8
    _thermal_spi.lsbfirst = False
    spi = SPI_Interface(_thermal_spi, xfer_size=160)

    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False, pin_factory=factory)
    mi48_reset_n    = DigitalOutputDevice("BCM23", active_high=False, initial_value=True, pin_factory=factory)

    class MI48_reset:
        def __init__(self, pin):
            self.pin = pin
        def __call__(self):
            self.pin.on(); time.sleep(0.000035)
            self.pin.off(); time.sleep(0.050)

    mi48 = MI48([i2c, spi], data_ready=mi48_data_ready,
                reset_handler=MI48_reset(pin=mi48_reset_n))
    mi48.set_fps(9)
    if int(mi48.fw_version[0]) >= 2:
        mi48.enable_filter(f1=True, f2=True, f3=False)
        mi48.set_offset_corr(0.0)
    mi48.start(stream=True, with_header=True)
    time.sleep(5)

    DISPLAY_SIZE = (640, 512)
    scale_x = DISPLAY_SIZE[0] / 80
    scale_y = DISPLAY_SIZE[1] / 62
    frame_count = 0

    while running:
        if hasattr(mi48, 'data_ready'):
            mi48.data_ready.wait_for_active()
        else:
            data_ready = False
            while not data_ready:
                time.sleep(0.01)
                data_ready = mi48.get_status() & DATA_READY

        try:
            with _spi0_lock:
                data, _ = mi48.read()
        except Exception:
            logger.exception("MI48 read error")
            continue
        if data is None:
            continue
        img = data_to_frame(data, mi48.fpa_shape)

        img_float = img.astype(np.float32)
        p2, p98 = np.percentile(img_float, 2), np.percentile(img_float, 98)
        if p98 <= p2:
            continue
        img8u = np.clip((img_float - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        img8u = cv_filter(img8u, parameters={'blur_ks': 3}, use_median=False, use_bilat=True, use_nlm=False)
        img8u = cv.flip(cv.rotate(img8u, cv.ROTATE_180), 1)

        frame_count += 1
        if frame_count < THERMAL_WARMUP:
            continue

        detections = detect_blobs(img8u)

        if detections:
            with state_lock:
                thermal_last_detection = time.time()

        display = cv.applyColorMap(img8u, cv.COLORMAP_JET)
        display = cv.resize(display, DISPLAY_SIZE, interpolation=cv.INTER_CUBIC)
        for (x1, y1, x2, y2) in detections:
            cv.rectangle(display,
                         (int(x1 * scale_x), int(y1 * scale_y)),
                         (int(x2 * scale_x), int(y2 * scale_y)),
                         (0, 255, 0), 2)
            cv.putText(display, "Person", (int(x1 * scale_x), int(y1 * scale_y) - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status = f"THERMAL: {len(detections)} detected" if detections else "THERMAL: searching..."
        color  = (0, 255, 0) if detections else (0, 0, 255)
        cv.putText(display, status, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if thermal_display_queue.qsize() == 0:
            thermal_display_queue.put(display)

    mi48.stop(stop_timeout=0.5)
    logger.info("Thermal stream stopped cleanly")

# ─────────────────────────────────────────────
# YOLO process (separate core)
# ─────────────────────────────────────────────
def rgb_detection_worker(in_q, out_q):
    from ultralytics import YOLO
    model = YOLO('/home/aurora/Drone-Machine-Learning/yolov8n.pt')
    while True:
        frame = in_q.get()
        if frame is None:
            break
        results = model(frame, classes=[0], verbose=False, imgsz=256)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            out_q.put([(list(map(float, b.xyxy[0])), float(b.conf[0])) for b in boxes])
        else:
            out_q.put([])

print("Loading YOLOv8n...")
detection_in_q  = multiprocessing.Queue(maxsize=1)
detection_out_q = multiprocessing.Queue(maxsize=1)
rgb_process = multiprocessing.Process(target=rgb_detection_worker, args=(detection_in_q, detection_out_q), daemon=True)
rgb_process.start()

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},
    controls={"FrameRate": 30, "AeEnable": True, "AwbEnable": True, "Brightness": 0.1}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

latest_rgb_detections = []

time.sleep(1)
thermal_thread = threading.Thread(target=thermal_stream, daemon=True)
thermal_thread.start()

def cleanup():
    global running
    running = False
    detection_in_q.put(None)
    rgb_process.terminate()
    picam2.stop()
    cv.destroyAllWindows()

def signal_handler(sig, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

cv.namedWindow('RGB Stream', cv.WINDOW_NORMAL)
cv.resizeWindow('RGB Stream', 1920, 1080)

print("AURORA running - Press 'q' to quit, 's' to save")
frame_count = 0

try:
    while True:
        frame = picam2.capture_array()
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        if detection_in_q.qsize() == 0 and frame_count % 3 == 0:
            detection_in_q.put(frame.copy())

        if not detection_out_q.empty():
            latest_rgb_detections = detection_out_q.get_nowait()
            if len(latest_rgb_detections) > 0:
                with state_lock:
                    rgb_last_detection = time.time()

        display = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        person_count = len(latest_rgb_detections)

        for (xyxy, conf) in latest_rgb_detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cv.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.putText(display, f"Person {conf:.2f}", (x1, y1 - 15),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        now = time.time()
        with state_lock:
            rgb_recent     = (now - rgb_last_detection)     < FUSION_WINDOW_SEC
            thermal_recent = (now - thermal_last_detection) < FUSION_WINDOW_SEC
        confirmed = rgb_recent and thermal_recent

        if confirmed:
            status = "!! CONFIRMED SURVIVOR !!"
            color  = (0, 255, 0)
            logger.info("CONFIRMED SURVIVOR — rgb_age=%.3fs thermal_age=%.3fs",
                        now - rgb_last_detection, now - thermal_last_detection)
            try:
                _survivor_event.set()
                logger.info("survivor event set successfully")
            except Exception:
                logger.exception("Failed to set survivor event")
        elif person_count > 0:
            status = "RGB: Person detected"
            color  = (0, 255, 255)
        else:
            status = "SEARCHING..."
            color  = (0, 0, 255)

        cv.putText(display, status, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv.putText(display, f"Frame: {frame.shape[1]}x{frame.shape[0]}",
                   (20, display.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('RGB Stream', display)

        if not thermal_display_queue.empty():
            cv.imshow('Thermal Stream', thermal_display_queue.get())

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"fusion_{frame_count}.jpg"
            cv.imwrite(fname, display)
            print(f"Saved {fname} - confirmed={confirmed}")
            frame_count += 1

except KeyboardInterrupt:
    pass
except Exception:
    logger.exception("Main loop crashed")
finally:
    cleanup()
    logging.shutdown()
    led_show([(0, 0, 0)] * NUM_LEDS)
    _led_spi.close()
