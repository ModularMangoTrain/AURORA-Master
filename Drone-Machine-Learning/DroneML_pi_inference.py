from ultralytics import YOLO
from picamera2 import Picamera2
import cv2 as cv
import time
import numpy as np
import threading
from queue import Queue
import spidev

# --- LED setup ---
NUM_LEDS = 144
_spi = spidev.SpiDev()
_spi.open(0, 0)
_spi.max_speed_hz = 3200000
_spi.mode = 0

def _encode_byte(byte):
    result = []
    for i in range(7, -1, -1):
        result.append(0b11100000 if byte & (1 << i) else 0b10000000)
    return result

def _led_show(pixels):
    data = []
    for r, g, b in pixels:
        data += _encode_byte(g) + _encode_byte(r) + _encode_byte(b)
    data += [0] * 10
    _spi.xfer2(data)

_led_active = threading.Event()

def _led_thread():
    while True:
        if _led_active.is_set():
            _led_show([(255, 220, 0)] * NUM_LEDS)
            time.sleep(0.15)
            _led_show([(0, 0, 0)] * NUM_LEDS)
            time.sleep(0.15)
        else:
            _led_show([(0, 0, 0)] * NUM_LEDS)
            time.sleep(0.1)

threading.Thread(target=_led_thread, daemon=True).start()
# --- end LED setup ---

# Load YOLOv8-nano model
print("Loading YOLOv8-nano model...")
model = YOLO('yolov8n.pt')

# Configure Pi Camera HD for maximum FPS at full resolution
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},  # Full HD resolution
    controls={
        "FrameRate": 30,           # Maximum FPS
        "AeEnable": True,          # Auto-exposure enabled
        "AwbEnable": True,         # Auto white balance
        "Brightness": 0.1          # Slight brightness boost
    }
)
picam2.configure(config)
picam2.start()

time.sleep(2)

# Debug: Check actual camera resolution
test_frame = picam2.capture_array()
print(f"Actual camera resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
print("Full HD Search and Rescue Mode - Press 'q' to quit, 's' to save")

frame_count = 0
detection_counter = 0
latest_detections = []

# Threading for non-blocking detection
detection_queue = Queue(maxsize=1)

def detection_worker():
    while True:
        frame = detection_queue.get()
        if frame is None:
            break
        
        # Run YOLO detection on full HD frame
        results = model(frame, classes=[0], verbose=False)
        global latest_detections
        latest_detections = results[0].boxes if results[0].boxes is not None else []
        detection_queue.task_done()

# Start detection thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

try:
    while True:
        frame = picam2.capture_array()
        
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        
        # Run detection on every frame
        detection_counter += 1
        if detection_queue.qsize() == 0:
            detection_queue.put(frame.copy())
        
        # Convert RGB to BGR for display
        display_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        # Use latest detections (from previous frames)
        person_count = len(latest_detections)
        _led_active.set() if person_count > 0 else _led_active.clear()
        
        # Draw bounding boxes using latest detections
        for box in latest_detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw box with thicker lines for HD
            cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label with larger font for HD
            label = f"Person {conf:.2f}"
            cv.putText(display_frame, label, (x1, y1-15), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Display count with larger text for HD
        status = f"SURVIVORS: {person_count}" if person_count > 0 else "SEARCHING..."
        color = (0, 255, 0) if person_count > 0 else (0, 0, 255)
        cv.putText(display_frame, status, (20, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        
        # Show actual frame dimensions
        info = f"Frame: {frame.shape[1]}x{frame.shape[0]} | Detection: Every frame"
        cv.putText(display_frame, info, (20, display_frame.shape[0]-30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create full-size window and display
        cv.namedWindow('Search and Rescue', cv.WINDOW_NORMAL)
        cv.resizeWindow('Search and Rescue', 1920, 1080)
        cv.imshow('Search and Rescue', display_frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"rescue_{frame_count}.jpg"
            cv.imwrite(filename, display_frame)
            print(f"Saved {filename} - {person_count} person(s) detected")
            frame_count += 1

except KeyboardInterrupt:
    pass
finally:
    detection_queue.put(None)  # Stop detection thread
    picam2.stop()
    cv.destroyAllWindows()
    _led_show([(0, 0, 0)] * NUM_LEDS)
    _spi.close()
