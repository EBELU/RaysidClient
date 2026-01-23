import asyncio
import time
import platform
import threading

import numpy as np

from bleak import BleakClient, BleakCharacteristicNotFoundError, BleakError
from bleak.exc import BleakDBusError

from .src import (
    logger,
    build_ping_packet,
    build_spectrum_request,
    build_tx2_packet,
    decode_cps_packet,
    decode_spectrum_packet,
    decode_status_packet,
    CurrentValuesPackage,
    SpectrumResult,
    StatusPackage,
    SpectrumAccumulator,
    two_bytes_to_int,
    unpack_value,
)   
    


SERVICE_UUID = "49535343-fe7d-4ae5-8fa9-9fafd205e455"
RX_UUID = "49535343-1e4d-4bd9-ba61-23c647249616"
TX_UUID = "49535343-8841-43f4-a8d4-ecbe34729bb3"

if platform.system() == "Windows":
    RX_UUID = RX_UUID.upper()
    TX_UUID = TX_UUID.upper()

_connect_lock = asyncio.Lock()  # ensures only one BLE connect/scan at a time

class RaysidClientAsync:
    def __init__(self, address):
        self._address = address
        self.name = ""
        self._client = BleakClient(address, timeout=15, disconnected_callback=self._on_disconnect)
        
        self._latest_cps: CurrentValuesPackage | None = None
        self._latest_status: StatusPackage | None = None
        
        self._spectrum_accumulator = SpectrumAccumulator()
        
        self._running = False
        self._disconnected = asyncio.Event()
        self._stopped = False
        
        self._tx_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._tx_task: asyncio.Task | None = None
        self._ping_task: asyncio.Task | None = None
        self.active_tab = 0
        
        self._spectrum_packet_buffer = None

        


    # ------------------ PUBLIC API ------------------
    @property
    def LatestRealTimeData(self):
        return self._latest_cps

    @property
    def LatestStatusData(self):
        return self._latest_status
    
    @property
    def LatestSpectrum(self):
        return self._spectrum_accumulator.snapshot()
    
    async def Reset(self, Erange = 2):
        Erange = np.uint8(Erange)
        """
        8 - switch to spectrum range 25..1000kev
        4 - switch to spectrum range 30..2000kev
        2 - switch to spectrum range 40..3500kev
        """
        if Erange not in [np.uint(2), np.uint8(4), np.uint(8)]:
            raise RuntimeError(f"Invalid energy range in reset, recieved range command {Erange}")

        data = bytearray(2)
        data[0] = 0x10              # command - clear spectrum
        data[1] = Erange
        await self.send_packet(data)

    # ------------------ CONTEXT MANAGER ------------------
    async def __aenter__(self):
        started_client = await self.start()
        return started_client
    
    async def start(self):
        success = False
        connection_attempts = 0
        while not success and connection_attempts < 5:
            connection_attempts += 1
            success = await self._connect()
            if not success:
                logger.warning(f"❌ Connection attempt {connection_attempts}/5 failed! Retrying in 3s...")
                await asyncio.sleep(3)
                if connection_attempts == 4:
                    raise RuntimeError("Connection failed!")

        
        self._running = True
        
        self._tx_task = asyncio.create_task(self._tx_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        logger.info("✅ Client started successfully")
        return self
    

    async def __aexit__(self, exc_type, exc, tb):                
        await self.stop()
        
    async def stop(self):
        self._stopped = True
        self._ping_task.cancel()
        self._tx_task.cancel()
        
        await asyncio.gather(
            self._ping_task,
            self._tx_task,
            return_exceptions=True
        )
        if self._client.is_connected:
            await self._client.stop_notify(RX_UUID)
            await self._client.disconnect()
            
        self._running = False

    # ------------------ PRIVATE METHODS ------------------

    def _on_disconnect(self, client):
        if self._running:
            logger.warning(f"❌ BLE device {self.device_name} disconnected!")
            if not self._stopped:
                asyncio.create_task(self._reconnect_loop())

        self._running = False
        self._disconnected.set()

        if hasattr(self, "_parent_on_disconnect") and self._parent_on_disconnect:
            self._parent_on_disconnect()


    async def _reconnect_loop(self):
        """Attempt to reconnect until successful or client is stopped."""
        max_attempts = 5  # 0 = infinite
        attempt = 0

        while not self._stopped:
            attempt += 1
            if max_attempts and attempt > max_attempts:
                logger.error("❌ Maximum reconnection attempts reached!")
                break

            logger.info(f"🔄 Attempting to reconnect to {self.device_name} (try {attempt})...")
            success = await self._connect()
            if success:
                logger.info(f"✅ Reconnected to {self.device_name}")
                self._running = True
                # Restart TX and ping loops
                self._tx_task = asyncio.create_task(self._tx_loop())
                self._ping_task = asyncio.create_task(self._ping_loop())
                break

            await asyncio.sleep(3)  # wait 3s before retry




    async def _start_notify(self, uuid, callback):
        """

        """

        try:
            await self._client.start_notify(uuid, callback)
            logger.info(f"✅ Notifications started")
            return  True
        except (BleakCharacteristicNotFoundError, BleakError) as e:
            logger.debug(f"❌ Could not start notifications on {uuid}")
            logger.debug(f"{str(e)}")
            return False

                    
    async def _connect(self):
        """Connect safely, retrying if BlueZ is busy"""
        async with _connect_lock:
            try:
                await self._client.connect()
                await asyncio.sleep(0.5) 

                success = await self._start_notify(RX_UUID, self._handle_notification)
                if not success:
                    await self._client.disconnect()
                    return False
                

                if isinstance(self._address, str):
                    self.device_name = self._address
                else:
                    self.device_name = self._address.name

                logger.info(f"✅ Connected to {self.device_name}")
                return True
            
            except BleakDBusError as e:
                if "InProgress" in str(e):
                    logger.warning(f"⚠️ BLE busy")
                    return False
                else:
                    return False

            
            
    async def _tx_loop(self):
        """
        Single-writer BLE transmit loop.
        """
        while self._running and self._client.is_connected:
            try:
                packet = await self._tx_queue.get()
                
                packet = build_tx2_packet(packet)

                await self._client.write_gatt_char(
                    TX_UUID,
                    packet,
                    response=True
                )
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ TX failed: {e}")

            
    async def send_packet(self, packet: bytes):
        await self._tx_queue.put(packet)


    
    async def _ping_loop(self):
        try:
            spect_packet = build_ping_packet(1)
            cps_packet = build_ping_packet(0)
            await self.send_packet(spect_packet)
            
            while self._running and self._client.is_connected:
                await self.send_packet(spect_packet)
                await asyncio.sleep(.7)

                await self.send_packet(cps_packet)
                logger.debug("Ping!")
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            pass


    # ------------------ NOTIFICATION HANDLER ------------------
    def _handle_notification(self, _, data: bytearray):
        if len(data) < 2:
            return
        pkt_type = data[1]

        if pkt_type == 0x17:
            cps = decode_cps_packet(np.frombuffer(data, np.uint8))
            if cps[0] != 0:
                error_code = cps[0]
                if error_code == 1:
                    logger.debug("⚠️  OVERLOAD — CPS not updated")
                elif error_code == 2:
                    logger.debug("⚠️  No CPS returned, unknown error")
                else:
                    raise RuntimeError("CPS decode failed, unknown error")
            if cps[1] > 0 and cps[2] > 0:
                logger.debug(f"📈 CPS: {cps[1]:.3f} | Dose rate: {cps[2]:.3f} uSv/h")

                self._latest_cps = CurrentValuesPackage(*cps[1:], time.time())
        elif pkt_type == 0x02:
            status = decode_status_packet(np.frombuffer(data, np.uint8))
            
            ERR            = 0
            UART_ERRORS    = 1
            BATTERY        = 2
            TEMPERATURE    = 3
            TEMP_OK        = 4
            CHARGING       = 5
            CHANNEL_FULL   = 6
            CH239          = 7
            AVG_CH239      = 8
            
            if status[ERR] != 0:
                raise RuntimeError("Status decode failed")
            
            # Log in a readable format
            logger.debug(
                f"🔋 Battery: {status[BATTERY]}% {'(charging)' if status[CHARGING] else ''} | "
                f"🌡 Temperature: {status[TEMPERATURE]:.1f}°C | Temp OK: {status[TEMP_OK]} | "
                f"Full channel: {status[CHANNEL_FULL]} | "
                f"239keV ch: {status[CH239]} avg: {status[AVG_CH239]} | "
                f"UART errors: {status[UART_ERRORS]}"
            )
            
            if status[UART_ERRORS] == 0:
                self._latest_status = StatusPackage(*status[BATTERY:], time.time())
                
        if pkt_type in (0x30, 0x31, 0x32) and len(data) > 7:
            if self._spectrum_packet_buffer:
                spectrum_result = decode_spectrum_packet(np.frombuffer(self._spectrum_packet_buffer, np.uint8))
                logger.debug("Decoding Spectrum")
                self._spectrum_accumulator.insert(*spectrum_result[1:])
                self._spectrum_packet_buffer = None
                logger.debug("Spectrum Decoded")
                error_code = spectrum_result[0]
                if error_code == 1:
                    logger.warning("Invalid spetrum packet received")
                elif error_code == 2:
                    logger.warning("Index error in X")
                elif error_code == 3:
                    logger.warning("Index error in pos")
            else:
                self._spectrum_packet_buffer = data
        elif self._spectrum_packet_buffer:
            try:
                self._spectrum_packet_buffer += (data)
            except BufferError:
                self._spectrum_packet_buffer = None
                

import asyncio
import threading
import traceback


class RaysidClient:
    def __init__(self, address):
        self._address = address

        self._loop = None
        self._thread = None
        self._client = None

        self._started = threading.Event()
        self._stopped = threading.Event()

        self.on_disconnect = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._thread:
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=False,  # ❗ DO NOT daemonize
        )
        self._thread.start()

        if not self._started.wait(timeout=10):
            raise RuntimeError("RaysidClient failed to start")

    def stop(self):
        if not self._loop:
            return

        async def shutdown():
            try:
                if self._client:
                    await self._client.stop()  # BLE disconnect
            except Exception:
                traceback.print_exc()
            finally:
                self._loop.stop()

        try:
            fut = asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
            fut.result(timeout=5)
        except Exception:
            traceback.print_exc()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self._client = None
        self._loop = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def LatestRealTimeData(self):
        return self._client.LatestRealTimeData if self._client else None

    @property
    def LatestStatusData(self):
        return self._client.LatestStatusData if self._client else None

    @property
    def LatestSpectrum(self):
        return self._client.LatestSpectrum if self._client else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        main_task = self._loop.create_task(self._start_client())

        try:
            self._loop.run_forever()
        finally:
            # Cancel remaining tasks cleanly
            for task in asyncio.all_tasks(self._loop):
                task.cancel()

            self._loop.run_until_complete(
                asyncio.gather(*asyncio.all_tasks(self._loop), return_exceptions=True)
            )

            self._loop.close()
            self._stopped.set()

    async def _start_client(self):
        self._client = RaysidClientAsync(self._address)
        self._client._parent_on_disconnect = self._handle_disconnect
        await self._client.start()
        self._started.set()

    def _handle_disconnect(self):
        if self.on_disconnect:
            self.on_disconnect()
            
    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
