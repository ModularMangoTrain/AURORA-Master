#!/usr/bin/env python3
"""
MI48 Thermal Camera with ML-Optimized Person Detection
Raspberry Pi 5 - Bookworm OS

Usage:
    sudo python3 thermal_detect.py              # Stream only
    sudo python3 thermal_detect.py -detect      # Stream with detection
    sudo python3 thermal_detect.py -model model.tflite  # Use ML model
    sudo python3 thermal_detect.py -collect data/  # Collect training data
"""

import sys
import os
import signal
from smbus2 import SMBus
from spidev import SpiDev
import argparse
import time
import numpy as np
import cv2 as cv
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from senxor.mi48 import MI48, DATA_READY
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface
from pathlib import Path
from collections import deque

# Configuration
I2C_CHANNEL = 1
I2C_ADDRESS = 0x40
SPI_BUS = 0
SPI_DEVICE = 1
SPI_MAX_SPEED_HZ = 31200000

def parse_args():
    parser = argparse.ArgumentParser(description='MI48 Thermal Camera with ML Person Detection')
    parser.add_argument('output', nargs='?', default=None, help='Output video file (optional)')
    parser.add_argument('-fps', type=int, default=9, help='FPS (default: 9)')
    parser.add_argument('-duration', type=int, default=0, help='Duration in seconds (0=manual)')
    parser.add_argument('-width', type=int, default=640, help='Width (default: 640)')
    parser.add_argument('-height', type=int, default=480, help='Height (default: 480)')
    parser.add_argument('-detect', action='store_true', help='Enable person detection')
    parser.add_argument('-model', type=str, default=None, help='TFLite model path')
    parser.add_argument('-collect', type=str, default=None, help='Collect data to directory')
    parser.add_argument('-conf', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('-fullscreen', action='store_true', help='Fullscreen HDMI output')
    return parser.parse_args()

def preprocess_for_ml(thermal_img, target_size=(96, 96)):
    """Preprocess thermal image for ML inference"""
    # Resize to model input size
    resized = cv.resize(thermal_img, target_size, interpolation=cv.INTER_LINEAR)
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    # Add batch and channel dimensions
    return np.expand_dims(normalized, axis=(0, -1))

def detect_person_threshold(thermal_img):
    """Fast threshold-based detection"""
    _, thresh = cv.threshold(thermal_img, 180, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    person_detected = False
    boxes = []
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 200:
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            if 0.5 < aspect_ratio < 3.0:
                person_detected = True
                boxes.append((x, y, w, h))
    
    return person_detected, boxes

class TFLiteDetector:
    """TensorFlow Lite model inference"""
    def __init__(self, model_path, conf_threshold=0.5):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.conf_threshold = conf_threshold
        self.input_shape = self.input_details[0]['shape'][1:3]
    
    def detect(self, thermal_img):
        """Run inference on thermal image"""
        # Preprocess
        input_data = preprocess_for_ml(thermal_img, tuple(self.input_shape))
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Parse results (assuming classification output)
        confidence = float(output[0]) if len(output.shape) == 1 else float(output.max())
        person_detected = confidence > self.conf_threshold
        
        return person_detected, confidence, []

class TemporalFilter:
    """Exponential moving average filter"""
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.prev_frame = None
    
    def filter(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.astype(np.float32)
            return frame
        filtered = self.alpha * frame + (1 - self.alpha) * self.prev_frame
        self.prev_frame = filtered
        return filtered.astype(frame.dtype)

class DataCollector:
    """Collect thermal data for ML training"""
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'person').mkdir(exist_ok=True)
        (self.output_dir / 'no_person').mkdir(exist_ok=True)
        self.count = {'person': 0, 'no_person': 0}
    
    def save(self, thermal_img, label):
        """Save thermal image with label"""
        subdir = 'person' if label else 'no_person'
        filename = f"{subdir}_{self.count[subdir]:05d}.npy"
        np.save(self.output_dir / subdir / filename, thermal_img)
        self.count[subdir] += 1
    
    def get_stats(self):
        return self.count

def main():
    args = parse_args()
    
    print("=" * 60)
    print("MI48 Thermal Camera with ML Person Detection")
    print("=" * 60)
    print(f"Mode: {'Recording' if args.output else 'Data Collection' if args.collect else 'Streaming'}")
    print(f"Detection: {'ML Model' if args.model else 'Threshold' if args.detect else 'Disabled'}")
    print(f"FPS: {args.fps}, Resolution: {args.width}x{args.height}")
    if args.model:
        print(f"Model: {args.model}, Confidence: {args.conf}")
    print("=" * 60)
    
    # Initialize hardware
    factory = LGPIOFactory(chip=4)
    i2c = I2C_Interface(SMBus(I2C_CHANNEL), I2C_ADDRESS)
    spi = SPI_Interface(SpiDev(SPI_BUS, SPI_DEVICE), xfer_size=160)
    spi.device.mode = 0
    spi.device.max_speed_hz = SPI_MAX_SPEED_HZ
    spi.device.bits_per_word = 8
    spi.device.lsbfirst = False
    
    mi48_data_ready = DigitalInputDevice("BCM24", pull_up=False, pin_factory=factory)
    mi48_reset_n = DigitalOutputDevice("BCM23", active_high=False, initial_value=True, pin_factory=factory)
    
    class MI48_reset:
        def __init__(self, pin):
            self.pin = pin
        def __call__(self):
            self.pin.on()
            time.sleep(0.000035)
            self.pin.off()
            time.sleep(0.050)
    
    # Initialize sensor
    mi48 = MI48([i2c, spi], data_ready=mi48_data_ready, reset_handler=MI48_reset(pin=mi48_reset_n))
    mi48.set_fps(min(args.fps, 9))
    
    # Enable all filtering
    if int(mi48.fw_version[0]) >= 2:
        mi48.enable_filter(f1=True, f2=True, f3=True)
        mi48.set_offset_corr(0.0)
    
    # Setup video writer
    out = None
    if args.output:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
        if not out.isOpened():
            print("ERROR: Could not open video writer")
            sys.exit(1)
    
    # Setup ML detector
    ml_detector = None
    if args.model:
        print(f"Loading model: {args.model}")
        ml_detector = TFLiteDetector(args.model, args.conf)
        print(f"Model loaded. Input shape: {ml_detector.input_shape}")
    
    # Setup data collector
    data_collector = None
    if args.collect:
        data_collector = DataCollector(args.collect)
        print(f"Data collection enabled: {args.collect}")
        print("Press 'p' for person, 'n' for no person")
    
    temp_filter = TemporalFilter(alpha=0.8)
    fps_buffer = deque(maxlen=30)
    
    # Setup display window
    window_name = 'MI48 Thermal Camera'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL if args.fullscreen else cv.WINDOW_AUTOSIZE)
    if args.fullscreen:
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    
    print("\nStarting...")
    mi48.start(stream=True, with_header=True)
    print(f"Ready! Press 'q' to quit, 'f' to toggle fullscreen\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
            
            if hasattr(mi48, 'data_ready'):
                mi48.data_ready.wait_for_active()
            
            data, _ = mi48.read()
            if data is None:
                continue
            
            # Process frame
            img = data_to_frame(data, mi48.fpa_shape).astype(np.float32)
            img = temp_filter.filter(img)
            
            # ML-optimized normalization (preserves thermal range)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_norm = ((img - img_min) * 255.0 / (img_max - img_min))
                img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
            else:
                img_norm = np.zeros_like(img, dtype=np.uint8)
            
            # Store raw thermal for ML
            thermal_raw = img_norm.copy()
            
            # Filter
            img_norm = cv_filter(img_norm, parameters={'blur_ks': 3}, 
                                use_median=False, use_bilat=True, use_nlm=False)
            
            # Colormap
            img_colored = cv.applyColorMap(img_norm, cv.COLORMAP_JET)
            img_resized = cv.resize(img_colored, (args.width, args.height), 
                                   interpolation=cv.INTER_LINEAR)
            
            # Detection
            if args.detect or ml_detector:
                if ml_detector:
                    # ML inference
                    person_detected, confidence, boxes = ml_detector.detect(thermal_raw)
                    status = f"PERSON {confidence:.2f}" if person_detected else f"CLEAR {1-confidence:.2f}"
                else:
                    # Threshold detection
                    person_detected, boxes = detect_person_threshold(img_norm)
                    status = "PERSON DETECTED" if person_detected else "No person"
                    confidence = 1.0 if person_detected else 0.0
                
                # Draw boxes
                for (x, y, w, h) in boxes:
                    scale_x = args.width / img_norm.shape[1]
                    scale_y = args.height / img_norm.shape[0]
                    x_scaled = int(x * scale_x)
                    y_scaled = int(y * scale_y)
                    w_scaled = int(w * scale_x)
                    h_scaled = int(h * scale_y)
                    
                    cv.rectangle(img_resized, (x_scaled, y_scaled), 
                               (x_scaled + w_scaled, y_scaled + h_scaled), 
                               (0, 255, 0), 2)
                
                color = (0, 255, 0) if person_detected else (0, 0, 255)
                cv.putText(img_resized, status, (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Data collection mode
            if data_collector:
                stats = data_collector.get_stats()
                cv.putText(img_resized, f"P:{stats['person']} N:{stats['no_person']}", 
                          (10, args.height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv.imshow(window_name, img_resized)
            
            if out:
                out.write(img_resized)
            
            frame_count += 1
            
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                current = cv.getWindowProperty(window_name, cv.WND_PROP_FULLSCREEN)
                if current == cv.WINDOW_FULLSCREEN:
                    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                else:
                    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            elif data_collector:
                if key == ord('p'):
                    data_collector.save(thermal_raw, True)
                    print(f"\nSaved: person ({data_collector.count['person']})")
                elif key == ord('n'):
                    data_collector.save(thermal_raw, False)
                    print(f"\nSaved: no_person ({data_collector.count['no_person']})")
            
            # FPS calculation
            fps_buffer.append(time.time())
            if len(fps_buffer) > 1:
                actual_fps = (len(fps_buffer) - 1) / (fps_buffer[-1] - fps_buffer[0])
            else:
                actual_fps = 0
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Frames: {frame_count}, FPS: {actual_fps:.1f}, Time: {elapsed:.1f}s", end='\r')
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        mi48.stop(stop_timeout=0.5)
        if out:
            out.release()
        cv.destroyAllWindows()
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\nComplete! Frames: {frame_count}, Duration: {elapsed:.1f}s, FPS: {actual_fps:.1f}")
        if args.output:
            print(f"Saved: {args.output}")
        if data_collector:
            stats = data_collector.get_stats()
            print(f"Data collected - Person: {stats['person']}, No person: {stats['no_person']}")

if __name__ == "__main__":
    main()
