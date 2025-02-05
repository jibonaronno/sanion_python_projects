import select, socket, sys, queue


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setblocking(0)
server.bind(('0.0.0.0', 5000))