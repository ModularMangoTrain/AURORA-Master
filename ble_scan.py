import asyncio
from bleak import BleakScanner
import json

OUTPUT_FILE = "/home/aurora/ble_devices.json"
devices = {}

def callback(device, advertisement_data):
    name = advertisement_data.local_name or "Unknown"
    devices[device.address] = {"name": name, "rssi": advertisement_data.rssi}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(list(devices.values())[-20:], f)
    print(f"[BLE] {name} | {device.address} | {advertisement_data.rssi}dBm")

async def main():
    scanner = BleakScanner(detection_callback=callback, adapter="hci1")
    await scanner.start()
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
