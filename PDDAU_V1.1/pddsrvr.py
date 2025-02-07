import socket
import select

class PddSrvr(object):
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def run_server(self, host='192.168.246.147', port=5000):
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
                    else:
                        # Existing connection has sent some data
                        try:
                            data = notified_socket.recv(1024)
                        except ConnectionResetError:
                            # Handle case where client disconnects abruptly
                            data = None

                        if data:
                            # Data received, for demonstration we simply echo it back to the client.
                            message = data.decode().strip()
                            print(f"Received message from {notified_socket.getpeername()}: {message}")
                            notified_socket.send(data)  # Echo back the received data
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

# if __name__ == '__main__':
#     try:
#         run_server()
#     except KeyboardInterrupt:
#         print("Server shutting down.")
