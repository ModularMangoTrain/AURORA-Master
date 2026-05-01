#!/usr/bin/env python3
"""
Dual Camera: Thermal (MI48) + RGB (Pi Camera)
Press 'q' to quit
"""

import time
import numpy as np
import cv2 as cv
from smbus2 import SMBus
from spidev import SpiDev
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory
from senxor.mi48 import MI48
from senxor.utils import data_to_frame, cv_filter
from senxor.interfaces import SPI_Interface, I2C_Interface
from picamera2 import Picamera2

# Thermal config
I2C_CHANNEL = 1
I2C_ADDRESS = 0x40
SPI_BUS = 0
SPI_DEVICE = 1

def main():
    print("Initializing cameras...")
    
    # Init thermal camera
    factory = LGPIOFactory(chip=4)
    i2c = I2C_Interface(SMBus(I2C_CHANNEL), I2C_ADDRESS)
    spi = SPI_Interface(SpiDev(SPI_BUS, SPI_DEVICE), xfer_size=160)
    spi.device.mode = 0
    spi.device.max_speed_hz = 31200000
    
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
    
    thermal = MI48([i2c, spi], data_ready=mi48_data_ready, reset_handler=MI48_reset(pin=mi48_reset_n))
    thermal.set_fps(9)
    if int(thermal.fw_version[0]) >= 2:
        thermal.enable_filter(f1=True, f2=True, f3=True)
    
    # Init RGB camera
    rgb = Picamera2()
    config = rgb.create_preview_configuration(main={"size": (640, 480)})
    rgb.configure(config)
    rgb.start()
    
    thermal.start(stream=True, with_header=True)
    print("Ready! Press 'q' to quit\n")
    
    thermal_resized = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_thermal = None
    global_min, global_max = None, None
    
    try:
        while True:
            # Get thermal frame
            if hasattr(thermal, 'data_ready'):
                thermal.data_ready.wait_for_active()
            
            data, _ = thermal.read()
            if data is not None:
                img = data_to_frame(data, thermal.fpa_shape).astype(np.float32)
                
                # Temporal smoothing
                if prev_thermal is not None:
                    img = 0.8 * img + 0.2 * prev_thermal
                prev_thermal = img.copy()
                
                # Global normalization (like stream_spi)
                if global_min is None:
                    global_min, global_max = img.min(), img.max()
                else:
                    global_min = min(global_min, img.min())
                    global_max = max(global_max, img.max())
                
                if global_max > global_min:
                    thermal_norm = ((img - global_min) * 255.0 / (global_max - global_min))
                    thermal_norm = np.clip(thermal_norm, 0, 255).astype(np.uint8)
                else:
                    thermal_norm = np.zeros_like(img, dtype=np.uint8)
                
                thermal_norm = cv_filter(thermal_norm, parameters={'blur_ks': 3}, 
                                        use_median=False, use_bilat=False, use_nlm=False)
                thermal_colored = cv.applyColorMap(thermal_norm, cv.COLORMAP_JET)
                thermal_resized = cv.resize(thermal_colored, (640, 480), interpolation=cv.INTER_CUBIC)
                thermal_resized = cv.flip(thermal_resized, -1)
            
            # Get RGB frame
            rgb_frame = rgb.capture_array()
            rgb_frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
            rgb_resized = cv.resize(rgb_frame, (640, 480))
            
            # Combine side by side
            combined = np.hstack([thermal_resized, rgb_resized])
            
            cv.imshow('Thermal + RGB', combined)
            
            if cv.waitKey(1) == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        thermal.stop()
        rgb.stop()
        cv.destroyAllWindows()
        print("\nDone!")

if __name__ == "__main__":
    main()
