# AURORA — Aerial Urban Rescue (for) Observation (and) Recovery Aid

AURORA is a drone-mounted search and rescue detection system built on a Raspberry Pi 5. It fuses RGB camera data with thermal infrared imaging to detect survivors, triggers LED alerts, and passively scans for nearby Bluetooth Low Energy devices. The system is designed to run fully autonomously on boot with no user interaction required.

Co-developed with Salman Aurmeeraly 

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [System Architecture](#system-architecture)
3. [How Everything Works](#how-everything-works)
   - [RGB Camera & YOLO Detection](#rgb-camera--yolo-detection)
   - [Thermal Camera (MI48)](#thermal-camera-mi48)
   - [Sensor Fusion](#sensor-fusion)
   - [LED Control](#led-control)
   - [BLE Scanning](#ble-scanning)
4. [Repository Structure](#repository-structure)
5. [Setting Up on a New Raspberry Pi](#setting-up-on-a-new-raspberry-pi)
   - [1. Flash Raspberry Pi OS](#1-flash-raspberry-pi-os)
   - [2. Enable Required Interfaces](#2-enable-required-interfaces)
   - [3. Install Nix](#3-install-nix)
   - [4. Clone the Repository](#4-clone-the-repository)
   - [5. Install pysenxor](#5-install-pysenxor)
   - [6. Set Up the Environment](#6-set-up-the-environment)
   - [7. Install Systemd Services](#7-install-systemd-services)
   - [8. Set CPU Governor](#8-set-cpu-governor)
6. [Running Manually](#running-manually)
7. [Debugging](#debugging)
8. [Hardware Wiring](#hardware-wiring)
9. [Known Issues & Notes](#known-issues--notes)

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Raspberry Pi 5 | 4GB or 8GB RAM |
| Raspberry Pi Camera Module | IMX477 (HQ Camera) |
| Meridian Innovation MI48 Thermal HAT | Mounted directly on GPIO header |
| WS2812B LED Strip | 144 LEDs, powered by external 5V PSU |
| USB Bluetooth Dongle | For BLE scanning (hci1) |
| Active cooling | Heatsink + fan strongly recommended — Pi runs hot under load |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CNN.py (Main Process)             │
│                                                     │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────┐  │
│  │ RGB Camera  │   │   Thermal    │   │   LED    │  │
│  │  Picamera2  │   │  MI48 Thread │   │  Thread  │  │
│  └──────┬──────┘   └──────┬───────┘   └────┬─────┘  │
│         │                 │                │        │
│  ┌──────▼──────┐          │         ┌──────▼─────┐  │
│  │ YOLO Process│          │         │  SPI0 CE0  │  │
│  │ (own core)  │          │         │  WS2812B   │  │
│  └──────┬──────┘          │         └────────────┘  │
│         │                 │                         │
│  ┌──────▼─────────────────▼──────┐                  │
│  │         Fusion Logic          │                  │
│  │  RGB + Thermal within 0.5s?   │                  │
│  │  → CONFIRMED SURVIVOR         │                  │
│  └───────────────────────────────┘                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              ble_scan.py (Separate Process)         │
│  Scans for BLE devices → writes to ble_devices.json │
└─────────────────────────────────────────────────────┘
```

---

## How Everything Works

### RGB Camera & YOLO Detection

The RGB camera (IMX477) captures frames at 1920x1080 at 30fps using `picamera2`. Every 3rd frame is sent to a separate process running YOLOv8n for person detection. Using a separate process (via `multiprocessing`) means YOLO runs on its own CPU core, leaving the main loop free to handle display and fusion logic without being blocked.

YOLO is configured to only detect class 0 (person) at an inference size of 256x256 for speed. When a person is detected, the timestamp `rgb_last_detection` is updated.

Bounding boxes and confidence scores are passed back from the YOLO process as plain Python lists (not YOLO objects) so they can be safely serialised across the process boundary.

### Thermal Camera (MI48)

The MI48 thermal camera is a HAT that sits directly on the GPIO header. It communicates over two interfaces simultaneously:
- **SPI0 CE1** (`/dev/spidev0.1`) — frame data at 3.9MHz
- **I2C bus 1** (`0x40`) — configuration and control
- **BCM24** — data ready interrupt pin
- **BCM23** — reset pin

The thermal thread reads 80x62 pixel frames from the MI48 at 9fps. Each frame goes through:
1. Percentile normalisation (2nd to 98th percentile mapped to 0-255)
2. Bilateral filter smoothing via `cv_filter`
3. 180° rotation and horizontal flip to correct for HAT orientation
4. Hot blob detection using contour analysis

Blob detection works by thresholding at the 85th percentile of pixel values (the hottest 15% of the frame), then finding contours and filtering by size (8-2000 pixels) and aspect ratio (0.3-6.0) to match human body shapes.

The first 20 frames are discarded as warmup to allow the sensor to stabilise.

A shared `_spi0_lock` is used between the thermal thread and the LED SPI to prevent simultaneous access to SPI0 which would cause crashes.

### Sensor Fusion

Fusion is simple and effective — a detection is only confirmed as a survivor if **both** the RGB camera and the thermal camera have detected a person within the last 0.5 seconds (`FUSION_WINDOW_SEC`). This eliminates false positives from either sensor alone.

```
confirmed = rgb_recent AND thermal_recent
```

When confirmed, the `_survivor_event` threading event is set, which triggers the LED flash sequence.

### LED Control

144 WS2812B LEDs are driven over **SPI0 CE0** (`/dev/spidev0.0`) at 3.2MHz. The WS2812B protocol is bit-banged over SPI by encoding each bit as a specific SPI byte pattern:
- Bit 1 → `0b11100000` (long high, short low)
- Bit 0 → `0b10000000` (short high, long low)

The LED thread runs continuously and has three states:
1. **Normal — Daytime**: Solid yellow `(255, 220, 0)` — based on a hardcoded sunrise/sunset table per month
2. **Normal — Night**: Solid red `(255, 0, 0)`
3. **Survivor confirmed**: Flashes the current day/night colour at low brightness `(10, 9, 0)` or `(10, 0, 0)` for 10 seconds

The brightness during flashing is intentionally kept very low to prevent excessive current draw from the LED strip which can cause voltage drops on the SPI lines and crash the Pi.

The LEDs require their own dedicated 5V power supply. Do not power them from the Pi's 5V pin.

### BLE Scanning

BLE scanning runs as a completely separate script (`ble_scan.py`) to avoid any performance impact on the main detection loop. It uses the `bleak` library with the USB Bluetooth dongle (`hci1`) to passively scan for nearby BLE advertisement packets.

Detected devices are written to `/home/aurora/ble_devices.json` (up to 20 most recent). The main `CNN.py` reads this file every 5 seconds and prints the device count to the terminal — a near-zero overhead operation.

---

## Repository Structure

```
AURORA-Master/
├── Drone-Machine-Learning/
│   ├── CNN.py                  ← Main script (cameras, YOLO, thermal, LEDs, fusion)
│   ├── flake.nix               ← Nix environment declaration
│   ├── requirements_project.txt← Python dependencies
│   ├── aurora.service          ← Systemd service for CNN.py
│   ├── aurora_ble.service      ← Systemd service for ble_scan.py
│   └── models/                 ← Model weights (via Git LFS)
│       ├── yolov8n.pt
│       └── spatial_person_detector_full.pth
├── pysenxor-master/            ← Modified Meridian Innovation senxor SDK
├── BLEAdverts/                 ← BLE scanning utilities
├── leds/                       ← Standalone LED test scripts
└── ble_scan.py                 ← Standalone BLE scanner
```

---

## Setting Up on a New Raspberry Pi

### Option A: Auto-Installer (Recommended)

SSH into the Pi and run:

```bash
curl -sSL https://raw.githubusercontent.com/ModularMangoTrain/AURORA-Master/main/Drone-Machine-Learning/install.py | python3
```

This will automatically:
- Update the system
- Enable SPI, I2C and Camera
- Clone the repository
- Create a virtual environment
- Install all dependencies including pysenxor
- Install and enable systemd services
- Set CPU governor to performance mode
- Prompt for a reboot

### Option B: Manual Setup

### 1. Flash Raspberry Pi OS

Flash the latest **Raspberry Pi OS (64-bit)** to an SD card using Raspberry Pi Imager. During setup:
- Set hostname to `AuroraPi5` (or your preference)
- Enable SSH
- Set username to `aurora`

### 2. Enable Required Interfaces

SSH into the Pi and run:

```bash
sudo raspi-config
```

Enable the following under **Interface Options**:
- SPI
- I2C
- Camera

Then add SPI1 support by editing `/boot/firmware/config.txt`:

```bash
sudo nano /boot/firmware/config.txt
```

Add at the bottom:
```
dtoverlay=spi1-1cs
```

Reboot:
```bash
sudo reboot
```

Verify SPI devices are available:
```bash
ls /dev/spidev*
# Should show: /dev/spidev0.0  /dev/spidev0.1  /dev/spidev1.0
```

### 3. Install Nix

```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
source /etc/profile.d/nix.sh
```

Enable flakes:
```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
sudo systemctl restart nix-daemon
```

### 4. Clone the Repository

```bash
git clone https://github.com/ModularMangoTrain/AURORA-Master.git
cd AURORA-Master
```

### 5. Install pysenxor

The modified pysenxor library must be installed manually as it contains custom fixes not in the official release:

```bash
cd pysenxor-master
pip install . --break-system-packages
cd ..
```

### 6. Set Up the Environment

```bash
cd Drone-Machine-Learning
nix develop
```

This will:
- Install all system dependencies (libcamera, bluez, i2c-tools, gcc)
- Create a Python virtual environment at `.venv`
- Install all Python packages from `requirements_project.txt`

The first run will take several minutes to download and build everything.

### 7. Install Systemd Services

Copy the service files and enable them so the system starts automatically on boot:

```bash
sudo cp Drone-Machine-Learning/aurora.service /etc/systemd/system/
sudo cp Drone-Machine-Learning/aurora_ble.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aurora aurora_ble
sudo systemctl start aurora aurora_ble
```

Verify they are running:
```bash
sudo systemctl status aurora aurora_ble
```

Pin BLE to its own CPU core to avoid performance impact:
```bash
sudo nano /etc/systemd/system/aurora_ble.service
```

Add `CPUAffinity=3` to the `[Service]` section:
```ini
[Service]
User=aurora
CPUAffinity=3
ExecStart=/home/aurora/drone_env/bin/python3 /home/aurora/ble_scan.py
Restart=on-failure
RestartSec=5
```

Pin the main script to cores 0-2:
```bash
sudo nano /etc/systemd/system/aurora.service
```

Add `CPUAffinity=0 1 2` to the `[Service]` section:
```ini
[Service]
User=aurora
CPUAffinity=0 1 2
Environment=DISPLAY=:0
WorkingDirectory=/home/aurora/Drone-Machine-Learning
ExecStart=/home/aurora/drone_env/bin/python3 /home/aurora/Drone-Machine-Learning/CNN.py
Restart=on-failure
RestartSec=5
```

Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart aurora aurora_ble
```

### 8. Set CPU Governor

Set the CPU to performance mode for maximum throughput:

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

To make this permanent across reboots:
```bash
sudo nano /etc/rc.local
```

Add before `exit 0`:
```bash
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
```

---

## Running Manually

If you want to run the scripts manually instead of via systemd:

```bash
# Activate the virtual environment
source /home/aurora/drone_env/bin/activate

# Run the main script
python3 /home/aurora/Drone-Machine-Learning/CNN.py

# Run BLE scanner in a separate terminal
python3 /home/aurora/ble_scan.py
```

To run with full logging output:
```bash
source /home/aurora/drone_env/bin/activate && python3 -u /home/aurora/Drone-Machine-Learning/CNN.py 2>&1 | tee /home/aurora/aurora_debug.log
```

---

## Debugging

View live logs from the systemd services:
```bash
# Main script logs
sudo journalctl -u aurora -f

# BLE scanner logs
sudo journalctl -u aurora_ble -f

# Debug log file
tail -f /home/aurora/aurora_debug.log
```

Check CPU temperature (watch for throttling above 80°C):
```bash
vcgencmd measure_temp
vcgencmd measure_clock arm
```

Check SPI devices are available:
```bash
ls /dev/spidev*
```

Check Bluetooth adapters:
```bash
hciconfig
```

---

## Hardware Wiring

### LED Strip (WS2812B)
| LED Strip | Raspberry Pi |
|-----------|-------------|
| Data      | GPIO10 (SPI0 MOSI, Pin 19) |
| GND       | GND (Pin 6) |
| 5V        | External 5V PSU (NOT Pi 5V pin) |

> The LED strip must be powered by a dedicated external 5V power supply. Powering from the Pi's 5V pin will cause voltage drops and system crashes when the LEDs draw full current.

### MI48 Thermal HAT
The MI48 sits directly on the 40-pin GPIO header. No additional wiring required. It uses:
- SPI0 CE1 for frame data
- I2C bus 1 for configuration
- BCM24 for data ready signal
- BCM23 for reset

### USB Bluetooth Dongle
Plug into any USB port. The system uses `hci1` (the USB dongle) for BLE scanning, leaving `hci0` (built-in) free.

---

## Known Issues & Notes

- **Thermal CRC errors on startup** — The MI48 logs CRC errors for the first few seconds while booting. This is normal and clears itself. The script waits 5 seconds after starting the sensor before reading frames.
- **LED brightness** — The flash brightness during survivor detection is intentionally kept low `(10, 9, 0)` to prevent current spikes crashing the SPI bus. The normal running brightness is full `(255, 220, 0)` / `(255, 0, 0)`.
- **Thermal throttling** — The Pi 5 runs hot under this workload. Without active cooling it will approach 80°C and throttle. A heatsink and fan are strongly recommended.
- **BLE lag** — Running BLE scanning in the same process as the main loop causes significant lag due to asyncio event loop interference. It is intentionally run as a separate service.
- **YOLO model download** — On first run, if `yolov8n.pt` is not found, ultralytics will automatically download it from the internet. Ensure the Pi has internet access on first boot.
