import socketio
import asyncio

# Create an asynchronous Socket.IO client
sio = socketio.AsyncClient()

# Event handler for connection
@sio.event
async def connect():
    print('Connected to the server')
    # Emit a message to the server after connecting
    await sio.emit('diagnosisEv', {'message': 'Hello from AsyncClient!'})

# Event handler for disconnection
@sio.event
async def disconnect():
    print('Disconnected from the server')

# Event handler for a specific event, e.g., 'my_async_event'
@sio.on('diagnosisEv')
async def on_my_async_event(data):
    print('Received "my_async_event" with data:', data)

# Function to connect to the server
async def connect_to_server():
    try:
        await sio.connect('http://localhost:8080')  # Replace with your server URL
        print('Successfully connected to the server')
    except Exception as e:
        print(f'Connection failed: {e}')

    # Wait for the connection to complete and keep the client alive
    await sio.wait()

# Run the client
if __name__ == '__main__':
    asyncio.run(connect_to_server())
