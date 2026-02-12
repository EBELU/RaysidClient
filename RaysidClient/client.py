import asyncio
import time
import platform
import threading
import contextlib
import logging

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
    decode_spectrum_meta_packet,
    CurrentValuesPackage,
    SpectrumResult,
    StatusPackage,
    SpectrumAccumulator,
)   


SERVICE_UUID = "49535343-fe7d-4ae5-8fa9-9fafd205e455"
RX_UUID = "49535343-1e4d-4bd9-ba61-23c647249616"
TX_UUID = "49535343-8841-43f4-a8d4-ecbe34729bb3"

if platform.system() == "Windows":
    RX_UUID = RX_UUID.upper()
    TX_UUID = TX_UUID.upper()

_connect_lock = asyncio.Lock()  # ensures only one BLE connect/scan at a time

class RaysidClientAsync:
    """
    Async implementation of the Raysid Client. Generally more stable and more responsive compared with the threaded version but async can be hassle.
    
    
    """
    def __init__(self, address, logger = None):
        
        self.logger = logger or logging.getLogger("RaysidClient")
        
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
        self._rx_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._rx_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None
        self._np_rx_buffer = np.empty(512, dtype=np.uint8) 
        self._ping_task: asyncio.Task | None = None
        self.active_tab = 1
        
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
    
    def reset(self, Erange = 4):
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
        self.send_packet(data)
        
    def set_active_tab(self, tab):
        self.active_tab = tab
        ping_packet = build_ping_packet(tab)
        self.send_packet(ping_packet)

    # ------------------ CONTEXT MANAGER ------------------
    async def __aenter__(self):
        started_client = await self.start()
        return started_client
    
    async def start(self):
        try:
            logger.info(f"📶 Connecting to {self._address}")
            success = False
            connection_attempts = 0
            while not success and connection_attempts < 6:
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
        except asyncio.CancelledError:
            logger.info("🛑 Client start cancelled — cleaning up")
            await self.stop()
            
            raise
    

    async def __aexit__(self, exc_type, exc, tb):                
        await self.stop()
        
    async def stop(self):
        self._stopped = True
        possible_tasks = [self._ping_task, self._tx_task, self._rx_task, self._reconnect_task]
        for task in possible_tasks:
            if task:
                task.cancel()
        
        await asyncio.gather(
            *possible_tasks,
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
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

        self._running = False
        self._disconnected.set()




    async def _reconnect_loop(self):
        """Attempt to reconnect until successful or client is stopped."""
        try:
            max_attempts = 5  # 0 = infinite
            attempt = 0
            
            restart_tasks = [
                task for task in (self._tx_task, self._rx_task, self._ping_task)
                if task and not task.done()
            ]

            for task in restart_tasks:
                task.cancel()
                
            for task in restart_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

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
                    self._rx_task = asyncio.create_task(self._packet_worker())
                    self._ping_task = asyncio.create_task(self._ping_loop())
                    break

                await asyncio.sleep(3)  # wait 3s before retry
        except asyncio.CancelledError:
            logger.info("🛑 Reconnect loop cancelled")
            await self.stop()
            raise


    async def _start_notify(self, uuid, callback):
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
                
                self._rx_queue = asyncio.Queue(maxsize=256)
                self._rx_task = asyncio.create_task(self._packet_worker())
                

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

            
    def send_packet(self, packet: bytes):
        """Put a byte packet in to tx_queue

        """
        self._tx_queue.put_nowait(packet)


    
    async def _ping_loop(self):
        """Pings the device every 10s to maintain connection
        """
        try:            
            while self._running and self._client.is_connected:
                ping_packet = build_ping_packet(self.active_tab)
                self.send_packet(ping_packet)
                logger.debug(f"Ping! {self.active_tab}")
                await asyncio.sleep(10)

        except asyncio.CancelledError:
            pass


    # ------------------ NOTIFICATION HANDLER ------------------
    def _handle_notification(self, _, data: bytearray):
        """
        BLE callback — must be extremely fast.
        Just copy bytes and enqueue.
        """

        try:
            self._rx_queue.put_nowait(bytes(data))
        except asyncio.QueueFull:
            logger.warning("⚠️ RX queue full — dropping packet")

        

    
    async def _packet_worker(self):
        """
        Background task that processes BLE packets.
        """
        try:
            while self._running and self._rx_task:
                data = await self._rx_queue.get()

                try:
                    self._process_packet(data)

                except Exception:
                    logger.exception("Packet processing failed")
        except asyncio.CancelledError:
            # cleanly exit loop
            return

    def _process_packet(self, data: bytes):
        if len(data) < 2:
            return

        pkt_type = data[1]
        buf = np.frombuffer(data, dtype=np.uint8)

        if pkt_type == 0x17:
            self._handle_cps_packet(buf)

        elif pkt_type == 0x02:
            self._handle_status_packet(buf)

        else:
            self._handle_spectrum_packet(buf)


    
    def _handle_cps_packet(self, data: bytes):
        cps = decode_cps_packet(data)

        error_code = cps[0]
        if error_code:
            if error_code == 1:
                logger.debug("⚠️ OVERLOAD — CPS not updated")
            elif error_code == 2:
                logger.debug("⚠️ No CPS returned, unknown error")
            else:
                raise RuntimeError("CPS decode failed")

        if cps[1] > 0 and cps[2] > 0:
            self._latest_cps = CurrentValuesPackage(
                cps[1], cps[2], time.time()
            )
            
    def _handle_status_packet(self, data: bytes):
        status = decode_status_packet(data)

        ERR, UART_ERRORS, BATTERY, TEMPERATURE, TEMP_OK, CHARGING, CHANNEL_FULL, CH239, AVG_CH239 = range(9)

        if status[ERR] != 0:
            raise RuntimeError("Status decode failed")
        # I dont like this :(
        if status[UART_ERRORS] == 0:
            self._latest_status = StatusPackage(
                *status[BATTERY:], time.time()
            )
    
    def _handle_spectrum_packet(self, data: bytes):
        if not hasattr(self, "_spectrum_chunks"):
            self._spectrum_chunks = []

        pkt_type = data[1]
        if pkt_type in (0x30, 0x31, 0x32) and len(data) > 7:
            if len(self._spectrum_chunks) > 0:
                packet = b"".join(self._spectrum_chunks)
                packet = np.frombuffer(packet, dtype=np.uint8)
                
                total_bytes = np.uint16(packet[0])

                if total_bytes == 0:
                    total_bytes = 256
                    

                spectrum_packet = None
                suffix_packet = None
                if len(packet) > total_bytes + 1:
                    suffix_packet = packet[total_bytes + 1:]
                    spectrum_packet = packet[:total_bytes + 1]

                else:
                    spectrum_packet = packet

                    
                cps = spect_meta = None
                if suffix_packet is not None:
                    if suffix_packet[1] == 0x17 and len(suffix_packet) == 13:

                        cps = suffix_packet
                    elif suffix_packet[1] == 0x01 and len(suffix_packet) == 21:
                        spect_meta = suffix_packet

                    elif len(suffix_packet) == 44:
                        spect_meta = suffix_packet[:-26]
                        cps = suffix_packet[-13:]
                    else:
                        logger.debug("Unknown suffix packet recieved")
                    
                    if cps is not None and cps[1] == 0x17:
                        self._handle_cps_packet(cps)
                        
                    if spect_meta is not None and spect_meta[1] == 0x01:
                        meta_data = decode_spectrum_meta_packet(suffix_packet)
                        if meta_data[0] == 0:
                            self._spectrum_accumulator.update_meta(*meta_data[1:])
                            
                        else:
                            logger.debug("Invalid length of meta packet")



                spectrum_result = decode_spectrum_packet(spectrum_packet)
                self._spectrum_chunks.clear()
                self._spectrum_chunks.append(data)
                error_code = spectrum_result[0]
                if error_code == 0:
                    self._spectrum_accumulator.insert(*spectrum_result[1:])
                elif error_code == 1:
                    logger.debug("Invalid spectrum packet")
                elif error_code == 2:
                    logger.debug("Index error in X")
                elif error_code == 3:
                    logger.debug("Index error in pos")
            else:
                self._spectrum_chunks.clear()
                self._spectrum_chunks.append(data)
        else:
            self._spectrum_chunks.append(data)
                


import threading
import traceback


class RaysidClient:
    """
    Threaded wrapper around RaysidClientAsync.
    
    Provides synchronous API access from another thread.
    """
    def __init__(self, address):
        self._address = address
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: RaysidClientAsync | None = None

        self._started = threading.Event()
        self._stopped = threading.Event()
        self.on_disconnect = None

    # -------------------- Public API --------------------
    def reset(self, Erange: int = 2):
        if not self._client or not self._loop:
            raise RuntimeError("Client not started")

        async def wrapper():
            await self._client.reset(Erange)

        fut = asyncio.run_coroutine_threadsafe(wrapper(), self._loop)
        return fut.result(timeout=5)

    def set_active_tab(self, tab: int):
        if not self._client or not self._loop:
            raise RuntimeError("Client not started")

        async def wrapper():
            await self._client.set_active_tab(tab)

        fut = asyncio.run_coroutine_threadsafe(wrapper(), self._loop)
        return fut.result(timeout=5)

    @property
    def LatestRealTimeData(self):
        return self._client.LatestRealTimeData if self._client else None

    @property
    def LatestStatusData(self):
        return self._client.LatestStatusData if self._client else None

    @property
    def LatestSpectrum(self):
        return self._client.LatestSpectrum if self._client else None

    # -------------------- Context Manager --------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # -------------------- Lifecycle --------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if not self._started.wait(timeout=10):
            raise RuntimeError("RaysidClient failed to start")

    def stop(self):
        if not self._loop or self._stopped.is_set():
            return

        async def shutdown():
            try:
                if self._client:
                    await self._client.stop()
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

        self._loop = None
        self._client = None
        self._stopped.set()

    # -------------------- Internal --------------------
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        main_task = self._loop.create_task(self._start_client())

        try:
            self._loop.run_forever()
        finally:
            for task in asyncio.all_tasks(self._loop):
                task.cancel()

            self._loop.run_until_complete(
                asyncio.gather(*asyncio.all_tasks(self._loop), return_exceptions=True)
            )
            self._loop.close()
            self._stopped.set()

    async def _start_client(self):
        self._client = RaysidClientAsync(self._address)
        # Wrap disconnect callback
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

if __name__ == "__main__":
    with RaysidClient("60:8A:10:32:15:43") as client:

        for i in range(40):
            print(client.LatestRealTimeData)
            
            time.sleep(0.5)
    