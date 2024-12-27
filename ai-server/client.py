import asyncio
import socketio

sio = socketio.AsyncSimpleClient()

async def connect():
    print('connection established')

async def my_message(data):
    print('message received with ', data)
    # await sio.emit('my response', {'response': 'my response'})

# @sio.event
# async def disconnect():
#     print('disconnected from server')

async def main():
    await sio.connect('http://localhost:8080')
    print("connection established")
    event = await sio.receive()
    print(f'received event: "{event[0]}" with arguments {event[1:]}')

    event = await sio.receive()
    print(f'received event: "{event[0]}" with arguments {event[1:]}')

if __name__ == '__main__':
    asyncio.run(main())