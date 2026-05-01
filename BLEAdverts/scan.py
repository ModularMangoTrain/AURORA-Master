#!/usr/bin/env python3
import asyncio
import csv
import os
import logging
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ble_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CSV_FILE = 'data.csv'

async def scan_callback(device: BLEDevice, advertisement_data: AdvertisementData):
    """
    BLE Scanner callback - must accept exactly 2 parameters
    """
    try:
        timestamp = datetime.now().isoformat()
        data = [
            timestamp,
            device.address,
            advertisement_data.rssi,
            advertisement_data.local_name or "Unknown",
            advertisement_data.manufacturer_data or ""
        ]
        
        # Write to CSV
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(data)
            f.flush()  # Flush immediately to ensure data is written
        
        logger.info(f"Device: {device.address} | {advertisement_data.local_name or 'No Name'} (RSSI: {advertisement_data.rssi})")
        
    except Exception as e:
        logger.error(f"Error in callback: {e}")

async def main():
    try:
        # Use hci0 for USB BLE adapter (check with hciconfig)
        adapter = "hci0"
        
        # Create scanner
        scanner = BleakScanner(
            detection_callback=scan_callback,
            adapter=adapter
        )
        
        # Start scanning
        logger.info(f"Starting BLE scan on adapter: {adapter}")
        await scanner.start()
        
        # Keep running until interrupted
        await asyncio.sleep(999999)
        
    except asyncio.CancelledError:
        logger.info("Scan cancelled")
    except Exception as e:
        logger.error(f"Error during scanning: {e}")
    finally:
        try:
            scanner.stop()
            logger.info("Scan stopped")
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program stopped by user (Ctrl+C)")
