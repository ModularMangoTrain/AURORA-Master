#!/usr/bin/env python3
import os
import subprocess
import sys

RED    = "\033[0;31m"
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
NC     = "\033[0m"

def log(msg):   print(f"{GREEN}[AURORA]{NC} {msg}")
def warn(msg):  print(f"{YELLOW}[WARN]{NC} {msg}")
def error(msg): print(f"{RED}[ERROR]{NC} {msg}"); sys.exit(1)

def run(cmd, check=True):
    return subprocess.run(cmd, shell=True, check=check)

def run_sudo(cmd):
    return run(f"sudo {cmd}")

print("""
  ╔═══════════════════════════════════════╗
  ║        AURORA Auto-Installer          ║
  ║   Autonomous Unmanned Reconnaissance  ║
  ╚═══════════════════════════════════════╝
""")

if os.geteuid() == 0:
    error("Do not run as root. Run as the aurora user.")

HOME        = os.path.expanduser("~")
INSTALL_DIR = os.path.join(HOME, "AURORA-Master")
VENV_DIR    = os.path.join(HOME, "drone_env")

# ─────────────────────────────────────────────
# Step 1: System update
# ─────────────────────────────────────────────
log("Updating system packages...")
run_sudo("apt update && sudo apt upgrade -y")

# ─────────────────────────────────────────────
# Step 2: System dependencies
# ─────────────────────────────────────────────
log("Installing system dependencies...")
packages = " ".join([
    "python3-pip", "python3-venv", "python3-dev",
    "git", "git-lfs", "i2c-tools", "python3-smbus",
    "libcamera-dev", "libcap-dev", "libcamera-apps",
    "bluez", "bluetooth", "libbluetooth-dev",
    "gcc", "cmake", "libopencv-dev", "python3-opencv",
    "libatlas-base-dev", "libhdf5-dev"
])
run_sudo(f"apt install -y {packages}")

# ─────────────────────────────────────────────
# Step 3: Enable interfaces
# ─────────────────────────────────────────────
log("Enabling SPI, I2C and Camera...")
run_sudo("raspi-config nonint do_spi 0")
run_sudo("raspi-config nonint do_i2c 0")
run_sudo("raspi-config nonint do_camera 0")

config_path = "/boot/firmware/config.txt"
with open(config_path, "r") as f:
    config = f.read()
if "dtoverlay=spi1-1cs" not in config:
    run_sudo(f"sh -c 'echo dtoverlay=spi1-1cs >> {config_path}'")
    log("SPI1 enabled")
else:
    warn("SPI1 already enabled, skipping")

# ─────────────────────────────────────────────
# Step 4: Clone repository
# ─────────────────────────────────────────────
if os.path.isdir(INSTALL_DIR):
    warn("AURORA-Master already exists, pulling latest...")
    run(f"git -C {INSTALL_DIR} pull")
else:
    log("Cloning AURORA-Master repository...")
    run(f"git clone https://github.com/ModularMangoTrain/AURORA-Master.git {INSTALL_DIR}")

run(f"git -C {INSTALL_DIR} lfs pull")

# ─────────────────────────────────────────────
# Step 5: Virtual environment
# ─────────────────────────────────────────────
log("Creating Python virtual environment...")
run(f"python3 -m venv {VENV_DIR} --system-site-packages")
pip = os.path.join(VENV_DIR, "bin", "pip")

# ─────────────────────────────────────────────
# Step 6: Install pysenxor
# ─────────────────────────────────────────────
log("Installing modified pysenxor...")
run(f"{pip} install {INSTALL_DIR}/pysenxor-master")

# ─────────────────────────────────────────────
# Step 7: Python dependencies
# ─────────────────────────────────────────────
log("Installing Python dependencies (this may take a while)...")
requirements = os.path.join(INSTALL_DIR, "Drone-Machine-Learning", "requirements_project.txt")
run(f"{pip} install -r {requirements}")

# ─────────────────────────────────────────────
# Step 8: Update hardcoded paths
# ─────────────────────────────────────────────
log("Updating paths...")
files_to_patch = [
    os.path.join(INSTALL_DIR, "Drone-Machine-Learning", "CNN.py"),
    os.path.join(INSTALL_DIR, "ble_scan.py"),
    os.path.join(INSTALL_DIR, "Drone-Machine-Learning", "aurora.service"),
    os.path.join(INSTALL_DIR, "Drone-Machine-Learning", "aurora_ble.service"),
]
for path in files_to_patch:
    if os.path.exists(path):
        with open(path, "r") as f:
            content = f.read()
        content = content.replace("/home/aurora/drone_env", VENV_DIR)
        content = content.replace("/home/aurora", HOME)
        with open(path, "w") as f:
            f.write(content)

# ─────────────────────────────────────────────
# Step 9: Systemd services
# ─────────────────────────────────────────────
log("Installing systemd services...")
run_sudo(f"cp {INSTALL_DIR}/Drone-Machine-Learning/aurora.service /etc/systemd/system/")
run_sudo(f"cp {INSTALL_DIR}/Drone-Machine-Learning/aurora_ble.service /etc/systemd/system/")
run_sudo("systemctl daemon-reload")
run_sudo("systemctl enable aurora aurora_ble")

# ─────────────────────────────────────────────
# Step 10: CPU governor
# ─────────────────────────────────────────────
log("Setting CPU governor to performance mode...")
run("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null", check=False)

rc_local = "/etc/rc.local"
with open(rc_local, "r") as f:
    rc = f.read()
governor_cmd = "for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $cpu; done"
if governor_cmd not in rc:
    rc = rc.replace("exit 0", f"{governor_cmd}\nexit 0")
    tmp = "/tmp/rc.local"
    with open(tmp, "w") as f:
        f.write(rc)
    run_sudo(f"cp {tmp} {rc_local}")
    run_sudo(f"chmod +x {rc_local}")

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print(f"""
{GREEN}╔═══════════════════════════════════════╗
║       Installation Complete!          ║
╚═══════════════════════════════════════╝{NC}

  A reboot is required to apply SPI/I2C/Camera changes.
  After reboot, AURORA will start automatically.

  To check status:  sudo systemctl status aurora
  To view logs:     journalctl -u aurora -f
""")

reboot = input("  Reboot now? (y/n): ").strip().lower()
if reboot == "y":
    run_sudo("reboot")
