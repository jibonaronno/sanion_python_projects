#!/usr/bin/env python3
"""
JAX-Optimized Spectrum Data Server - FIXED VERSION
High-performance batch processing for sine wave generation and streaming.
*** FIXES SKIPPED SECONDS AND JAX CONCRETIZATION ERRORS ***
"""

import socket
import struct
import select
import os
import time
import sys
import threading  # Add explicit threading import

# JAX imports with fallback and configuration
try:
    # Set JAX configuration for better error messages
    import os

    os.environ.setdefault('JAX_TRACEBACK_FILTERING', 'off')  # Show full tracebacks

    import jax
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
    print("JAX successfully imported - using GPU/CPU optimized processing")

    # Warm up JAX with a simple operation to catch early issues
    try:
        test_array = jnp.array([1, 2, 3])
        _ = jnp.sum(test_array)
        print("JAX basic operations verified")
    except Exception as e:
        print(f"JAX basic operation failed: {e}")
        JAX_AVAILABLE = False

except ImportError:
    print("JAX not available - falling back to NumPy")
    import numpy as jnp

    JAX_AVAILABLE = False


    # Create mock jit decorator for fallback
    def jit(func):
        return func
except Exception as e:
    print(f"JAX import failed with error: {e}")
    print("Falling back to NumPy")
    import numpy as jnp

    JAX_AVAILABLE = False


    # Create mock jit decorator for fallback
    def jit(func):
        return func

# PyQt5 imports
try:
    from PyQt5.QtCore import QThread, QElapsedTimer, pyqtSignal, pyqtSlot

    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt5 not available - using threading fallback")
    PYQT_AVAILABLE = False
    import threading

# Define the format strings:
HEADER_FORMAT = "<BBh"

NUM_CHANNEL = 4
NUM_SAMPLES = 1024
PACKETS_PER_SECOND = 60  # 60 Hz
SAMPLES_PER_SECOND = NUM_SAMPLES * PACKETS_PER_SECOND  # 61,440 samples per second

MIN_VAL = 0
MAX_VAL = 8000

GOING_DOWN = 12
GOING_UP = 21


# FIXED JAX-optimized sine wave generation (no concretization errors)
@jit
def generate_single_sine_packet_jax(max_value, min_value=0):
    """
    Generate a single sine wave packet using JAX.
    FIXED: Uses static shape to avoid concretization errors.
    """
    # Generate angles for one complete cycle (FIXED: static shape)
    angles = jnp.linspace(0, 2 * jnp.pi, NUM_SAMPLES, endpoint=False)

    # Generate sine wave
    sine_wave = jnp.sin(angles)

    # Scale to desired range [min_value, max_value]
    sine_wave_scaled = ((sine_wave + 1) / 2) * (max_value - min_value) + min_value

    # Convert to uint16
    sine_wave_uint16 = sine_wave_scaled.astype(jnp.uint16)

    return sine_wave_uint16


# Use static_argnums for the shape parameters
@jax.jit
def generate_sine_wave_batch_static(amplitudes):
    """
    FIXED: Generate a batch of sine wave packets with different amplitudes.
    Uses vmap for vectorization without dynamic shapes.
    """

    def single_packet(amp):
        return generate_single_sine_packet_jax(amp, MIN_VAL)

    # Use vmap to vectorize over amplitudes
    batch_fn = jax.vmap(single_packet)
    return batch_fn(amplitudes)


class FastDataGenerator:
    """
    FIXED: JAX-optimized data generator with buffering to prevent skipped seconds.
    """

    def __init__(self, buffer_size=3):
        self.current_batch_index = 0
        self.batch_buffer = []  # Buffer to store pre-generated batches
        self.buffer_size = buffer_size  # Number of batches to keep in buffer
        self.generation_lock = threading.Lock() if 'threading' in sys.modules else None

    def generate_second_batch(self, amplitude_start, amplitude_end):
        """
        FIXED: Generate 1 second worth of data (60 packets) with varying amplitude.
        Now optimized to avoid blocking the main streaming loop.
        """
        # Generate amplitude sweep for this second
        amplitudes = jnp.linspace(amplitude_start, amplitude_end, PACKETS_PER_SECOND)

        # Use the optimized batch generation
        if JAX_AVAILABLE:
            try:
                # Try to use the vectorized approach for better performance
                batch_data = generate_sine_wave_batch_static(amplitudes)
            except Exception as e:
                print(f"Vectorized generation failed, using fallback: {e}")
                # Fallback to individual packet generation
                batch_data = []
                for i, amp in enumerate(amplitudes):
                    sine_data = generate_single_sine_packet_jax(float(amp), MIN_VAL)
                    batch_data.append(sine_data)
                batch_data = jnp.stack(batch_data)
        else:
            # NumPy fallback
            batch_data = []
            for i, amp in enumerate(amplitudes):
                angles = jnp.linspace(0, 2 * jnp.pi, NUM_SAMPLES, endpoint=False)
                sine_wave = jnp.sin(angles)
                sine_wave_scaled = ((sine_wave + 1) / 2) * (float(amp) - MIN_VAL) + MIN_VAL
                sine_wave_uint16 = sine_wave_scaled.astype(jnp.uint16)
                batch_data.append(sine_wave_uint16)
            batch_data = jnp.stack(batch_data)

        return batch_data, amplitudes

    def get_buffered_batch(self, amplitude_start, amplitude_end):
        """
        FIXED: Get a batch from buffer or generate if needed.
        This is non-blocking and returns immediately.
        """
        if self.generation_lock:
            with self.generation_lock:
                if len(self.batch_buffer) > 0:
                    return self.batch_buffer.pop(0)
        elif len(self.batch_buffer) > 0:
            return self.batch_buffer.pop(0)

        # If no buffer available, generate immediately (blocking)
        return self.generate_second_batch(amplitude_start, amplitude_end)

    def pregenerate_batches(self, amplitude_start, amplitude_end, amplitude_direction):
        """
        FIXED: Pre-generate batches for the buffer to avoid streaming interruptions.
        This should be called in a separate thread.
        """
        try:
            current_amp = amplitude_start
            direction = amplitude_direction

            while len(self.batch_buffer) < self.buffer_size:
                # Calculate next amplitude
                if direction == GOING_DOWN:
                    next_amp = max(MIN_VAL + 400, current_amp - 200 * PACKETS_PER_SECOND)
                    if next_amp <= MIN_VAL + 400:
                        direction = GOING_UP
                else:  # GOING_UP
                    next_amp = min(MAX_VAL, current_amp + 200 * PACKETS_PER_SECOND)
                    if next_amp >= MAX_VAL:
                        direction = GOING_DOWN

                # Generate batch
                batch_data, amplitudes = self.generate_second_batch(current_amp, next_amp)

                # Add to buffer
                if self.generation_lock:
                    with self.generation_lock:
                        self.batch_buffer.append((batch_data, amplitudes, next_amp, direction))
                else:
                    self.batch_buffer.append((batch_data, amplitudes, next_amp, direction))

                current_amp = next_amp

        except Exception as e:
            print(f"Batch pregeneration error: {e}")


class SpectrumPacketHeader:
    def __init__(self, msg_id, msg_type, body_len):
        self.msg_id = msg_id
        self.msg_type = msg_type
        self.body_len = body_len

    def to_bytes(self):
        return struct.pack("<BBH", self.msg_id, self.msg_type, self.body_len)


class MsgPdBody:
    def __init__(self, ch_idx=0, EventAmpTh1=0, EventAmpTh2=0, EventPpsTh=0, data=None):
        self.ch_idx = ch_idx
        self.EventAmpTh1 = EventAmpTh1
        self.EventAmpTh2 = EventAmpTh2
        self.EventPpsTh = EventPpsTh
        self.data = data if data is not None else bytearray(NUM_SAMPLES * 2)

    def to_bytes(self):
        return struct.pack("<BBBB", self.ch_idx, self.EventAmpTh1, self.EventAmpTh2, self.EventPpsTh) + self.data


class FastMsgPdFullPacket:
    def __init__(self, msg_type, sine_wave_data):
        """
        FIXED: Create a packet from pre-generated JAX sine wave data.
        Enhanced error handling and data validation.
        """
        self.header = SpectrumPacketHeader(0x01, msg_type, NUM_CHANNEL * (NUM_SAMPLES * 2 + 4))

        # Convert JAX array to little-endian bytes
        if sine_wave_data is not None:
            try:
                # Convert to numpy for byte operations
                if JAX_AVAILABLE and hasattr(sine_wave_data, 'shape'):
                    # Handle JAX array
                    sine_wave_np = jnp.array(sine_wave_data)
                    # Ensure it's the right shape and type
                    if len(sine_wave_np.shape) == 1 and sine_wave_np.shape[0] == NUM_SAMPLES:
                        sine_wave_np = sine_wave_np.astype(jnp.uint16)
                        # Convert to little-endian bytes
                        self.sine_wave = sine_wave_np.astype('<u2').tobytes()
                    else:
                        print(f"Warning: Unexpected sine wave shape: {sine_wave_np.shape}")
                        # Fallback to zeros
                        self.sine_wave = bytearray(NUM_SAMPLES * 2)
                else:
                    # Handle numpy array or other array-like
                    import numpy as np
                    sine_wave_np = np.array(sine_wave_data)
                    if len(sine_wave_np.shape) == 1 and sine_wave_np.shape[0] == NUM_SAMPLES:
                        sine_wave_np = sine_wave_np.astype(np.uint16)
                        self.sine_wave = sine_wave_np.astype('<u2').tobytes()
                    else:
                        print(f"Warning: Unexpected sine wave shape: {sine_wave_np.shape}")
                        self.sine_wave = bytearray(NUM_SAMPLES * 2)
            except Exception as e:
                print(f"Error converting sine wave data: {e}")
                # Fallback to zeros
                self.sine_wave = bytearray(NUM_SAMPLES * 2)
        else:
            # Fallback for header-only packets
            self.sine_wave = bytearray(NUM_SAMPLES * 2)

        self.data = [MsgPdBody(data=self.sine_wave) for _ in range(NUM_CHANNEL)]

    def to_bytes(self):
        packet = self.header.to_bytes()
        for body in self.data:
            packet += body.to_bytes()
        return packet

    def header_only_bytes(self):
        return self.header.to_bytes()


# Server thread class - compatible with both PyQt5 and threading
if PYQT_AVAILABLE:
    class FastServerThread(QThread):
        received = pyqtSignal(str)

        def __init__(self, host='localhost', port=5000, parent=None):
            super().__init__(parent)
            self._init_server(host, port)

        def emit_signal(self, message):
            self.received.emit(message)

        def sleep_ms(self, ms):
            self.usleep(ms)

        def sleep_sec(self, sec):
            self.sleep(sec)
else:
    class FastServerThread(threading.Thread):
        def __init__(self, host='localhost', port=5000, parent=None):
            super().__init__()
            self._init_server(host, port)
            self.daemon = True

        def emit_signal(self, message):
            print(f"[SERVER] {message}")

        def sleep_ms(self, ms):
            time.sleep(ms / 1000000.0)  # Convert microseconds to seconds

        def sleep_sec(self, sec):
            time.sleep(sec)

        def stop(self):
            self.running = False

        def isInterruptionRequested(self):
            return not self.running


# FIXED: Add common initialization with precision timing and buffering
def _init_server(self, host, port):
    self.host = host
    self.port = port
    self.clients = []
    self.running = True
    self.data_generator = FastDataGenerator()

    # Pre-generate initial data batch
    self.current_amplitude = MAX_VAL
    self.amplitude_direction = GOING_DOWN
    self.current_batch = None
    self.batch_index = 0

    # FIXED: Timing precision improvements
    self.packet_count = 0
    self.start_time = None
    self.target_interval = 1.0 / PACKETS_PER_SECOND  # 16.67ms for 60Hz

    # Batch generation thread
    self.batch_generation_thread = None
    self.stop_batch_generation = False

    # Generate first batch with error handling
    try:
        self._generate_next_batch()
        print("Initial batch generated successfully")

        # FIXED: Start batch pre-generation thread (inline to avoid method binding issues)
        if 'threading' in sys.modules:
            def pregeneration_worker():
                print("Batch pre-generation worker started")
                while not self.stop_batch_generation and self.running:
                    try:
                        # Only generate if buffer is not full
                        if len(self.data_generator.batch_buffer) < self.data_generator.buffer_size:
                            self.data_generator.pregenerate_batches(
                                self.current_amplitude,
                                self.current_amplitude,  # Will be calculated in pregenerate_batches
                                self.amplitude_direction
                            )
                        else:
                            # Buffer is full, wait a bit
                            time.sleep(0.5)
                    except Exception as e:
                        print(f"Batch pregeneration thread error: {e}")
                        time.sleep(1)  # Wait longer on error
                print("Batch pre-generation worker stopped")

            self.batch_generation_thread = threading.Thread(target=pregeneration_worker, daemon=True)
            self.batch_generation_thread.start()
            print("Batch pre-generation thread started")

    except Exception as e:
        print(f"Warning: Initial batch generation failed: {e}")
        print("Will generate batches on-demand")
        self.current_batch = None


# Monkey patch the common init
FastServerThread._init_server = _init_server


def _generate_next_batch(self):
    """FIXED: Generate the next 1-second batch of data with non-blocking buffered approach."""
    amplitude_start = self.current_amplitude

    # Calculate amplitude change for this second
    if self.amplitude_direction == GOING_DOWN:
        amplitude_end = max(MIN_VAL + 400, self.current_amplitude - 200 * PACKETS_PER_SECOND)
        if amplitude_end <= MIN_VAL + 400:
            self.amplitude_direction = GOING_UP
    else:  # GOING_UP
        amplitude_end = min(MAX_VAL, self.current_amplitude + 200 * PACKETS_PER_SECOND)
        if amplitude_end >= MAX_VAL:
            self.amplitude_direction = GOING_DOWN

    # Try to get batch from buffer first (non-blocking)
    try:
        if hasattr(self.data_generator, 'get_buffered_batch'):
            buffered_result = self.data_generator.get_buffered_batch(amplitude_start, amplitude_end)
            if buffered_result and len(buffered_result) == 4:
                # Using pre-generated buffered batch
                self.current_batch, amplitudes, self.current_amplitude, self.amplitude_direction = buffered_result
                self.batch_index = 0
                self.emit_signal(f"Using buffered batch: amp {amplitude_start:.0f} -> {self.current_amplitude:.0f}")
                return

        # Fallback to immediate generation (blocking)
        self.current_batch, amplitudes = self.data_generator.generate_second_batch(
            amplitude_start, amplitude_end
        )
        self.current_amplitude = amplitude_end
        self.batch_index = 0
        self.emit_signal(f"Generated immediate batch: amp {amplitude_start:.0f} -> {amplitude_end:.0f}")

    except Exception as e:
        print(f"Error generating batch: {e}")
        self.emit_signal(f"Batch generation error: {e}")

        # Fallback: generate a simple batch with fixed amplitude
        try:
            print("Attempting fallback batch generation...")
            if JAX_AVAILABLE:
                # Simple fallback - single amplitude for all packets
                fixed_amp = (amplitude_start + amplitude_end) / 2
                batch_data = []
                for i in range(PACKETS_PER_SECOND):
                    packet = generate_single_sine_packet_jax(fixed_amp, MIN_VAL)
                    batch_data.append(packet)
                self.current_batch = jnp.stack(batch_data)
                self.current_amplitude = amplitude_end
                self.batch_index = 0
                self.emit_signal(f"Fallback batch generated with fixed amplitude: {fixed_amp:.0f}")
            else:
                # NumPy fallback
                import numpy as np
                fixed_amp = (amplitude_start + amplitude_end) / 2
                batch_data = []
                for i in range(PACKETS_PER_SECOND):
                    angles = np.linspace(0, 2 * np.pi, NUM_SAMPLES, endpoint=False)
                    sine_wave = np.sin(angles)
                    sine_wave_scaled = ((sine_wave + 1) / 2) * (fixed_amp - MIN_VAL) + MIN_VAL
                    sine_wave_uint16 = sine_wave_scaled.astype(np.uint16)
                    batch_data.append(sine_wave_uint16)
                self.current_batch = np.stack(batch_data)
                self.current_amplitude = amplitude_end
                self.batch_index = 0
                self.emit_signal(f"NumPy fallback batch generated: {fixed_amp:.0f}")

        except Exception as fallback_error:
            print(f"Fallback batch generation also failed: {fallback_error}")
            self.emit_signal(f"Critical: All batch generation methods failed")
            self.current_batch = None


def unpackReceivedData(self, data):
    if len(data) >= 4:
        unpacked = struct.unpack(HEADER_FORMAT, data)
        header = {
            "msg_id": unpacked[0],
            "msg_type": unpacked[1],
            "body_len": unpacked[2]
        }
        return header
    else:
        return None


def run(self):
    # Create server socket
    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 65536)
    self.server_socket.bind((self.host, self.port))
    self.server_socket.listen(5)
    self.server_socket.setblocking(False)

    # Create start packet
    start_packet = FastMsgPdFullPacket(0x11, None)
    packet_start = start_packet.header_only_bytes()

    sockets = [self.server_socket]
    self.emit_signal(f"Fast JAX server listening on {self.host}:{self.port}")
    self.emit_signal(f"showplot")

    while self.running and not self.isInterruptionRequested():
        try:
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
            rlist, wlist, exceptional = select.select(sockets, [], sockets, 0.01)
        except Exception as e:
            self.emit_signal(f"Select error: {e}")
            continue

        for sock in rlist:
            if sock is self.server_socket:
                # New client connection
                try:
                    client_socket, client_address = self.server_socket.accept()
                    client_socket.setblocking(False)
                    sockets.append(client_socket)
                    self.clients.append(client_socket)
                    self.emit_signal(f"New connection from {client_address}")
                except Exception as e:
                    self.emit_signal(f"Accept error: {e}")
            else:
                # Data from existing client
                try:
                    data = sock.recv(1024)
                    if data:
                        peer = sock.getpeername()
                        text = data.decode(errors="replace")
                        self.emit_signal(f"Data from {peer}: {text}")

                        if self.unpackReceivedData(data) is not None:
                            header = self.unpackReceivedData(data)
                            print(header)
                            self.send_sample_data(packet_start)

                        print("Start Sending Data Stream")
                        self.sleep_ms(10000)
                        print("Waiting 10 seconds")
                        self.sleep_sec(10)

                        # FIXED: Main streaming loop with precise timing and non-blocking batch generation
                        print("Starting high-precision 60Hz streaming...")
                        self.start_time = time.time()
                        self.packet_count = 0

                        while True:
                            try:
                                loop_start = time.time()

                                # Check if we need a new batch (non-blocking)
                                if self.batch_index >= PACKETS_PER_SECOND or self.current_batch is None:
                                    batch_start = time.time()
                                    self._generate_next_batch()
                                    batch_time = (time.time() - batch_start) * 1000
                                    if batch_time > 5:  # Log if batch generation takes >5ms
                                        print(f"Batch generation took {batch_time:.1f}ms")

                                # Check if we have valid batch data
                                if self.current_batch is None:
                                    self.emit_signal("No batch data available, waiting...")
                                    self.sleep_ms(1000)  # Wait 1ms before retrying (not blocking)
                                    continue

                                # Get current packet from batch
                                current_sine_data = self.current_batch[self.batch_index]

                                # Create packet with pre-generated data
                                packet_data = FastMsgPdFullPacket(0x03, current_sine_data)
                                packet_bytes = packet_data.to_bytes()

                                # Send the packet
                                send_start = time.time()
                                self.send_sample_data(packet_bytes)
                                send_time = (time.time() - send_start) * 1000

                                self.batch_index += 1
                                self.packet_count += 1

                                # FIXED: Monitor performance (inline to avoid method binding issues)
                                if not hasattr(self, 'performance_monitor'):
                                    self.performance_monitor = {
                                        'last_second': int(time.time()),
                                        'packets_this_second': 0,
                                        'total_packets': 0,
                                        'skipped_seconds': 0,
                                        'start_time': int(time.time())
                                    }

                                current_second = int(time.time())
                                self.performance_monitor['total_packets'] += 1

                                if current_second != self.performance_monitor['last_second']:
                                    # New second
                                    packets_last_second = self.performance_monitor['packets_this_second']

                                    if packets_last_second < 55:  # Allow some tolerance (should be 60)
                                        self.performance_monitor['skipped_seconds'] += 1
                                        self.emit_signal(
                                            f"WARNING: Only {packets_last_second} packets in last second (expected 60)")
                                        print(
                                            f"PERFORMANCE WARNING: Skipped second detected - only {packets_last_second} packets")

                                    # Log performance every 10 seconds
                                    if current_second % 10 == 0:
                                        total_time = max(1, current_second - self.performance_monitor['start_time'])
                                        avg_rate = self.performance_monitor['total_packets'] / total_time
                                        self.emit_signal(
                                            f"Performance: {avg_rate:.1f} Hz avg, {self.performance_monitor['skipped_seconds']} skipped seconds")

                                    # Reset for new second
                                    self.performance_monitor['last_second'] = current_second
                                    self.performance_monitor['packets_this_second'] = 1
                                else:
                                    self.performance_monitor['packets_this_second'] += 1

                                # FIXED: Precise timing calculation for 60Hz
                                target_time = self.start_time + (self.packet_count * self.target_interval)
                                current_time = time.time()
                                sleep_time = target_time - current_time

                                # Log timing issues
                                if self.packet_count % 300 == 0:  # Every 5 seconds
                                    actual_rate = self.packet_count / (current_time - self.start_time)
                                    print(
                                        f"Streaming: {actual_rate:.1f} Hz (target: 60 Hz), batch gen: {batch_time:.1f}ms, send: {send_time:.1f}ms")

                                # Sleep for precise timing
                                if sleep_time > 0:
                                    # Convert to microseconds for sleep_ms
                                    sleep_us = int(sleep_time * 1000000)
                                    if sleep_us > 1000:  # Only sleep if >1ms
                                        self.sleep_ms(sleep_us)
                                else:
                                    # We're running behind schedule
                                    if sleep_time < -0.005:  # More than 5ms behind
                                        print(f"Warning: Running {-sleep_time * 1000:.1f}ms behind schedule")

                                # Additional small delay to prevent CPU spinning
                                if sleep_time <= 0:
                                    self.sleep_ms(100)  # 0.1ms minimum

                            except Exception as e:
                                print(f"Streaming error: {e}")
                                self.emit_signal(f"Streaming error: {e}")
                                break
                    else:
                        # Connection closed
                        peer = sock.getpeername()
                        self.emit_signal(f"Connection closed by {peer}")
                        sockets.remove(sock)
                        if sock in self.clients:
                            self.clients.remove(sock)
                        sock.close()
                except Exception as e:
                    self.emit_signal(f"Receive error: {e}")
                    if sock in sockets:
                        sockets.remove(sock)
                    if sock in self.clients:
                        self.clients.remove(sock)
                    sock.close()


def send_sample_data(self, _data):
    for client in self.clients.copy():
        try:
            client.sendall(_data)
        except Exception as e:
            print(f"Error sending to {client.getpeername()}: {e}")
            try:
                client.close()
            except Exception:
                pass
            self.clients.remove(client)


# Monkey patch the methods (cleaned up - removed problematic method references)
FastServerThread._init_server = _init_server
FastServerThread._generate_next_batch = _generate_next_batch
FastServerThread.unpackReceivedData = unpackReceivedData
FastServerThread.run = run
FastServerThread.send_sample_data = send_sample_data

# Add PyQt5 specific methods with proper cleanup
if PYQT_AVAILABLE:
    def stop_pyqt(self):
        """Stop the server thread and cleanup batch generation."""
        self.running = False
        self.stop_batch_generation = True

        # Wait for batch generation thread to finish
        if hasattr(self, 'batch_generation_thread') and self.batch_generation_thread:
            try:
                self.batch_generation_thread.join(timeout=2.0)
                print("Batch generation thread stopped")
            except Exception as e:
                print(f"Error stopping batch generation thread: {e}")

        self.wait()


    FastServerThread.stop = stop_pyqt
else:
    def stop_threading(self):
        """Stop the server thread and cleanup batch generation."""
        self.running = False
        self.stop_batch_generation = True

        # Wait for batch generation thread to finish
        if hasattr(self, 'batch_generation_thread') and self.batch_generation_thread:
            try:
                self.batch_generation_thread.join(timeout=2.0)
                print("Batch generation thread stopped")
            except Exception as e:
                print(f"Error stopping batch generation thread: {e}")


    FastServerThread.stop = stop_threading


# FIXED: Benchmark function with proper error handling
def benchmark_data_generation():
    """Compare JAX vs NumPy performance for batch generation."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping benchmark")
        return

    import numpy as np
    import time

    print("Benchmarking data generation performance...")

    # Test parameters
    num_tests = 10
    packets_per_second = 60
    samples_per_packet = 1024

    # JAX benchmark - warm up first
    print("Warming up JAX...")
    try:
        # Warm up with single packet generation
        _ = generate_single_sine_packet_jax(MAX_VAL, MIN_VAL)

        # Test batch generation
        amplitudes = jnp.linspace(MIN_VAL + 400, MAX_VAL, packets_per_second)
        _ = generate_sine_wave_batch_static(amplitudes)

        start_time = time.time()
        for _ in range(num_tests):
            amplitudes = jnp.linspace(MIN_VAL + 400, MAX_VAL, packets_per_second)
            batch_data = generate_sine_wave_batch_static(amplitudes)
        jax_time = (time.time() - start_time) / num_tests

    except Exception as e:
        print(f"JAX vectorized benchmark failed: {e}")
        print("Falling back to individual packet benchmark...")

        # Fallback benchmark with individual packets
        _ = generate_single_sine_packet_jax(MAX_VAL, MIN_VAL)

        start_time = time.time()
        for _ in range(num_tests):
            batch_data = []
            for i in range(packets_per_second):
                amp = MIN_VAL + 400 + (MAX_VAL - MIN_VAL - 400) * i / packets_per_second
                packet = generate_single_sine_packet_jax(amp, MIN_VAL)
                batch_data.append(packet)
            batch_data = jnp.stack(batch_data)
        jax_time = (time.time() - start_time) / num_tests

    # NumPy benchmark (original method)
    def generate_sine_wave_numpy(max_value):
        angles = np.linspace(0, 2 * np.pi, samples_per_packet, endpoint=False)
        sine_wave = np.sin(angles)
        sine_wave_scaled = ((sine_wave + 1) / 2) * (max_value - MIN_VAL) + MIN_VAL
        return sine_wave_scaled.astype(np.uint16)

    start_time = time.time()
    for _ in range(num_tests):
        for i in range(packets_per_second):
            amp = MIN_VAL + 400 + (MAX_VAL - MIN_VAL - 400) * i / packets_per_second
            _ = generate_sine_wave_numpy(amp)
    numpy_time = (time.time() - start_time) / num_tests

    print(f"JAX batch generation: {jax_time * 1000:.2f} ms per second of data")
    print(f"NumPy individual packets: {numpy_time * 1000:.2f} ms per second of data")
    if jax_time > 0:
        print(f"Speed improvement: {numpy_time / jax_time:.1f}x faster with JAX")
    else:
        print("JAX performance measurement unclear")


if __name__ == '__main__':
    # Show JAX configuration
    print("=" * 60)
    print("JAX SPECTRUM SERVER CONFIGURATION")
    print("=" * 60)
    print(f"JAX Available: {JAX_AVAILABLE}")
    if JAX_AVAILABLE:
        try:
            print(f"JAX Backend: {jax.default_backend()}")
            print(f"JAX Devices: {jax.devices()}")
        except Exception as e:
            print(f"JAX info error: {e}")
    print(f"PyQt5 Available: {PYQT_AVAILABLE}")
    print("=" * 60)

    # Run benchmark
    benchmark_data_generation()

    # Test server
    print("\nStarting test server...")
    server = FastServerThread(host='localhost', port=5001)
    server.start()

    try:
        if PYQT_AVAILABLE:
            server.wait()
        else:
            server.join()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop()