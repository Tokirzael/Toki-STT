import queue
import threading
import time

from pythonosc import udp_client

from .logger import logger

# VRChat OSC addresses
CHATBOX_INPUT = "/chatbox/input"
CHATBOX_TYPING = "/chatbox/typing"


class OSCController:
    def __init__(self, ip="127.0.0.1", port=9000):
        self.ip = ip
        self.port = port
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        self.last_sent_time = 0
        # Minimum spacing between chatbox messages (seconds)
        self.min_interval = 1.5
        # VRChat truncates very long OSC chat messages; split at this length
        self.chunk_size = 144
        # Pause between chunks of a single long message (seconds)
        self.chunk_delay = 5.0

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info(f"OSC Controller started on {self.ip}:{self.port}")

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
            self.worker_thread = None

    def update_settings(self, ip, port):
        self.ip = ip
        self.port = port
        self.client = udp_client.SimpleUDPClient(ip, port)

    def send_message(self, address, value, immediate=False):
        if immediate:
            # Bypass queue for urgent things like the typing indicator
            try:
                self.client.send_message(address, value)
            except Exception as e:
                logger.error(f"OSC Send Error: {e}")
        else:
            self.queue.put((address, value))

    def _sleep(self, seconds):
        """Interruptible sleep so stop() is never blocked by a long wait."""
        end = time.time() + seconds
        while self.running:
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))

    def _send(self, address, value):
        self.client.send_message(address, value)
        self.last_sent_time = time.time()

    def _worker(self):
        # Plain FIFO: every queued message is delivered in order.
        # (The old "fold to newest partial" logic could drop messages that
        # arrived while the worker was sleeping between chunks.)
        while self.running:
            try:
                address, value = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if address == CHATBOX_INPUT and isinstance(value, str):
                # Rate limit: keep chatbox messages spaced out
                elapsed = time.time() - self.last_sent_time
                if elapsed < self.min_interval:
                    self._sleep(self.min_interval - elapsed)

                # Split long messages so VRChat doesn't truncate them
                chunks = [value[i:i + self.chunk_size]
                          for i in range(0, len(value), self.chunk_size)] or [""]
                for i, chunk in enumerate(chunks):
                    if i > 0:
                        self._sleep(self.chunk_delay)
                    try:
                        self._send(address, [chunk, True])
                    except Exception as e:
                        logger.error(f"OSC Worker Error (Chunk {i}): {e}")
            else:
                try:
                    self._send(address, value)
                except Exception as e:
                    logger.error(f"OSC Worker Error: {e}")

            self.queue.task_done()