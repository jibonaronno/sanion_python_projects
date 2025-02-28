import socket
import struct

import select
import os, time

# Define the format strings:
# '<' specifies little-endian. Adjust if you need big-endian (use '>').
HEADER_FORMAT = "<BBh"          # msg_id (B), msg_type (B), body_len (h)

import sys
from PyQt5.QtCore import QThread, QElapsedTimer, pyqtSignal, pyqtSlot

class ServerThread(QThread):
    received = pyqtSignal(str)

    def __init__(self, host='localhost', port=5000, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.clients = []  # List to keep track of connected client sockets
        self.random_data = os.urandom(256)
        self.running = True

    def unpackReceivedData(self, data):
        if len(data) >= 4:
            unpacked = struct.unpack(HEADER_FORMAT, data)
            header = {
                "msg_id":   unpacked[0],
                "msg_type": unpacked[1],
                "body_len": unpacked[2]
            }
            return header
        else:
            return None

    def run(self):
        # Create server socket and set up non-blocking mode
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.setblocking(False)
        received_data = b''

        # List of sockets to monitor: start with the server socket
        sockets = [self.server_socket]
        self.received.emit(f"Server listening on {self.host}:{self.port}")

        while self.running and not self.isInterruptionRequested():
            try:
                # Use a timeout so we can check self.running regularly
                # for sock in sockets:
                #     if sock.fileno() == -1:
                #         sockets.remove(sock)
                time.sleep(1)
                rlist, wlist, exceptional = select.select(sockets, [], sockets)
            except Exception as e:
                self.received.emit(f"Select error: {e}")
                continue

            for sock in rlist:
                if sock is self.server_socket:
                    # New client connection
                    try:
                        client_socket, client_address = self.server_socket.accept()
                        client_socket.setblocking(False)
                        sockets.append(client_socket)
                        self.clients.append(client_socket)
                        self.received.emit(f"New connection from {client_address}")
                    except Exception as e:
                        self.received.emit(f"Accept error: {e}")
                else:
                    # Data from an existing client
                    try:
                        data = sock.recv(1024)
                        if data:
                            # Emit the received data (decoded to text)
                            peer = sock.getpeername()
                            text = data.decode(errors="replace")
                            self.received.emit(f"Data from {peer}: {text}")
                            print("Start Sending Data Stream")
                            while 1:
                                for i in range(0, len(self.random_data), 2):
                                    two_bytes = data[i:i + 2]
                                    self.send_sample_data(two_bytes)
                                    # self.usleep(1953)
                                    self.usleep(217)
                        else:
                            # No data: client has closed connection
                            peer = sock.getpeername()
                            self.received.emit(f"Connection closed by {peer}")
                            sockets.remove(sock)
                            if sock in self.clients:
                                self.clients.remove(sock)
                            sock.close()
                    except Exception as e:
                        self.received.emit(f"Receive error: {e}")
                        if sock in sockets:
                            sockets.remove(sock)
                        if sock in self.clients:
                            self.clients.remove(sock)
                        sock.close()

                # Also process any sockets with errors (if needed)
                # (For brevity, exceptional conditions are not handled separately here.)

                # Clean up sockets when stopping
            # for sock in sockets:
            #     sock.close()
            # self.received.emit("Server stopped.")

    @pyqtSlot()
    def stop(self):
        """Stop the server thread."""
        self.running = False
        self.wait()  # Wait for thread to exit

    @pyqtSlot()
    def send_sample_data(self, _data):
        """Send sample data to all connected clients."""
        sample = "Sample data from server."
        # Convert the sample text to bytes.
        data = sample.encode()
        # Send to each client socket
        for client in self.clients.copy():
            try:
                client.sendall(_data)
            except Exception as e:
                # self.received.emit(f"Error sending to {client.getpeername()}: {e}")
                print(f"Error sending to {client.getpeername()}: {e}")
                try:
                    client.close()
                except Exception as e:
                    print(f"Error client.close : {e}")
                    pass
                self.clients.remove(client)
        # self.received.emit("Sent sample data to clients.")

class WorkerThread(QThread):
    # Signal to emit each time the interval has elapsed (we send the elapsed microseconds)
    tick = pyqtSignal(int)

    def run(self):
        timer = QElapsedTimer()
        timer.start()
        last_ns = timer.nsecsElapsed()  # nanoseconds elapsed at last tick
        interval_ns = 976 * 1000         # convert 976 microseconds to nanoseconds

        while not self.isInterruptionRequested():
            now_ns = timer.nsecsElapsed()
            if now_ns - last_ns >= interval_ns:
                # Emit the tick signal with the current elapsed time in microseconds
                self.tick.emit(int(now_ns / 1000))
                last_ns = now_ns
            # Yield a bit to avoid hogging the CPU entirely.
            # usleep takes microseconds as argument.
            self.usleep(10)

class PddSrvr(object):
    def __init__(self, stop_event, send_samples):
        super().__init__()
        self.stop_event = stop_event
        self.send_samples = send_samples
        self.samples128 = False

    def setSamples128(self, samples128):
        self.samples128 = samples128

    def run_server(self, host='192.168.246.13', port=5000):
        # Create a TCP/IP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow to reuse the address
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to the host and port
        server_socket.bind((host, port))
        # Listen for incoming connections
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        # Set the server socket to non-blocking mode
        server_socket.setblocking(False)

        # List of sockets to monitor for incoming data
        sockets_list = [server_socket]

        random_data = os.urandom(16384)

        if self.samples128:
            random_data = os.urandom(256)

        client_socket = None

        try:
            while not self.stop_event.is_set():
                # Use select to get the list of sockets ready for reading, writing or with errors.
                readable, writable, exceptional = select.select(sockets_list, [], sockets_list)

                for notified_socket in readable:
                    # If the notified socket is the server socket, it means a new connection is incoming.
                    if notified_socket == server_socket:
                        client_socket, client_address = server_socket.accept()
                        print(f"Accepted new connection from {client_address}")
                        client_socket.setblocking(False)
                        sockets_list.append(client_socket)
                        print(str(len(sockets_list)) + " Number Of Sockets \n")
                    else:
                        # Existing connection has sent some data
                        try:
                            data = notified_socket.recv(1024)
                        except ConnectionResetError:
                            # Handle case where client disconnects abruptly
                            data = None

                        if data:
                            # Data received, for demonstration we simply echo it back to the client.
                            # message = data.decode().strip()
                            # print(f"Received message from {notified_socket.getpeername()}: {message}")
                            print(f"Received message from {notified_socket.getpeername()}: ")
                            # notified_socket.send(data)  # Echo back the received data
                            # if self.send_samples.is_set():
                            print("Waiting 11 Sec Before Streaming \n")
                            time.sleep(11)
                            while 1:
                                print("Sending 1024 Samples \n")
                                client_socket.sendall(random_data)
                                time.sleep(1)
                        else:
                            # No data means the client gracefully closed the connection.
                            print(f"Connection closed from {notified_socket.getpeername()}")
                            sockets_list.remove(notified_socket)
                            notified_socket.close()

                # Handle exceptional conditions (if any socket errors occur)
                for notified_socket in exceptional:
                    print(f"Handling exceptional condition for {notified_socket.getpeername()}")
                    sockets_list.remove(notified_socket)
                    notified_socket.close()



        finally:
            print("Shutting down server...")
            # Close all remaining sockets
            for sock in sockets_list:
                sock.close()

class WorkerThread(QThread):
    # Signal to emit each time the interval has elapsed (we send the elapsed microseconds)
    tick = pyqtSignal(int)

    def run(self):
        timer = QElapsedTimer()
        timer.start()
        last_ns = timer.nsecsElapsed()  # nanoseconds elapsed at last tick
        interval_ns = 976 * 1000         # convert 976 microseconds to nanoseconds

        while not self.isInterruptionRequested():
            now_ns = timer.nsecsElapsed()
            if now_ns - last_ns >= interval_ns:
                # Emit the tick signal with the current elapsed time in microseconds
                self.tick.emit(int(now_ns / 1000))
                last_ns = now_ns
            # Yield a bit to avoid hogging the CPU entirely.
            # usleep takes microseconds as argument.
            self.usleep(10)

# if __name__ == '__main__':
#     try:
#         run_server()
#     except KeyboardInterrupt:
#         print("Server shutting down.")
