from .src.logger import logger
from bleak import BleakScanner

async def find_device_by_name(name: str, timeout=10):
    """
    Scan for BLE devices and return the first one matching the given name.
    """
    logger.info(f"🔎 Scanning for BLE devices named '{name}'...")

    def match_name(device, advertisement_data):
        return device.name and device.name.startswith(name)

    device = await BleakScanner.find_device_by_filter(match_name, timeout=timeout)
    
    if device:
        logger.info(f"✅ Found device '{device.name}' ({device.address})")
    else:
        logger.error(f"❌ Device '{name}' not found after {timeout} seconds.")
    return device