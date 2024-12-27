# import asyncio

# from rstream import Consumer, amqp_decoder, AMQPMessage

# # Create a consumer

# # More like a callback

# def on_message(msg: AMQPMessage):
#   print('Got message: {}'.format(msg.body))

# async def main():
#     consumer = Consumer(
#       host='localhost',
#       port=5552,
#       username='guest',
#       password='guest',
#       sasl_configuration_mechanism='PLAIN'
#   )

#     # Assuming consumer is an instance of Consumer
#     await consumer.start()
#     await consumer.subscribe('mystream', on_message, decoder=amqp_decoder)
#     await consumer.run()

# # Python 3.7+
# asyncio.run(main())

import pika
import torch
import os 
from SamDecoder import SAM_Decoder
from modeling import ImageEncoderViT
from torch.nn import functional as F
import helpers as helpers
import json
from torchvision.io import read_image
from torchvision import transforms,datasets
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import csv
from datetime import datetime
import asyncio
import numpy as np
import functools
from typing import  List
import concurrent.futures
import socketio
from aiohttp import web
from pika.adapters.asyncio_connection import AsyncioConnection

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)


@sio.event
def connect(sid, environ):
    print("connect ", sid)

@sio.event
async def chat_message(sid, data):
    print("message ", data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

async def handle(request):
    print("Abc")

app.add_routes([
    web.get('/', handle)])

sam_decoder:List[SAM_Decoder]
DEVICE=None
tasks=[]
countModel=2
loop=asyncio.new_event_loop()

import threading
def create_ram_drive(size_mb, drive_letter):
    # Ensure imdisk is installed and available in PATH
    os.system(f'imdisk -a -t vm -s {size_mb}M -m {drive_letter}: -p "/fs:NTFS /q /y"')

# Example usage
create_ram_drive(1024, 'R')
def load_model():
    global sam_decoder
    global countModel
    global DEVICE
    sam_decoder=[]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    PATH= './sam_classification_model_'+ str(7) +'_6.pth'
    torch.cuda.empty_cache() 
    DEVICE = torch.device('cuda:0')
    # sam_decoder = SAM_Decoder(sam_encoder = ImageEncoderViT)
    # sam_decoder.load_state_dict(torch.load('./sam_classification_model_7_6.pth'))
    # sam_decoder = sam_decoder.to(DEVICE)
    for x in range(countModel):
        sam_decoder2=SAM_Decoder(sam_encoder=ImageEncoderViT)
        sam_decoder2.load_state_dict(torch.load('./sam_classification_model_7_6.pth'))
        sam_decoder2=sam_decoder2.to(DEVICE)
        # sam_decoder.eval()
        sam_decoder2.eval()
        sam_decoder.append(sam_decoder2)
    print(sam_decoder)
    
load_model()
def sync(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        global loop
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(f(*args, **kwargs))

    return wrapper

class SingleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('jpg', 'jpeg', 'png', 'bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path)
        except:
            print("image is not found")
        # print(image)
        if self.transform:
            image = self.transform(image)
        return image  # Returning the image and its path

def preprocess( x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    img_size=264
    # Pad
    h, w = x.shape[-2:]
    # print(h,w)
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



def start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    loop.run_forever()
    print(loop)

   
async def async_draw_image(body, pictureId):
    loop = asyncio.get_event_loop()
    # print(body)
    result = await loop.run_in_executor(None, helpers.DrawImage, body, pictureId)
    return result
async def run_tasks():
    global tasks
    # print(tasks)
    tasks_to_run = tasks[:64]
    # print(tasks_to_run)
    tasks = tasks[64:]
    
    await asyncio.gather(*tasks_to_run)
async def run_tasksModel(tasksModel):
    tasks_to_run = tasksModel[:2]
    tasksModel = tasksModel[2:]
    
    await asyncio.gather(*tasks_to_run)

def run_event_loop(loop, coro):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coro)
    loop.close()

def receive_file():
    global countModel
    exchange_name='file_exchange'
    queue_name='file_queue'
    # Establish a connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    # Declare a queue (if it doesn't already exist)
    channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key="")
    batchedInput1=[]
    batchedInput2=[]
    counter=0
    pictureId=0
    tasksModel=[]
    flagModel=False
    valueDivider=32* countModel
    transform = transforms.Compose([
        transforms.Resize((264, 264)),
        transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    writerArray=[]
    # for i in range(countModel):
    current_date = datetime.now().strftime("%Y-%m-%d")
    csv_file_name = f'./results/predictions_{current_date}_${5}.csv'

    csv_file=open(csv_file_name, mode='a', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['Timestamp','FileName','SensorId', 'Prediction'])
    writerArray.append(writer) # Write header

    async def modelWrapper(decoder, batchedInput, index2):
        with torch.no_grad():
            multplier = index2 * 32
            input_images = torch.stack([transform(Image.open(f"R:\\file_{multplier + i}.jpg")) for i in range(32)], dim=0).to(DEVICE)
            sensorIds = [x['sensorId'] for x in batchedInput]
            fileNames = [x['fileName'] for x in batchedInput]
            try:
                outputs = decoder(input_images)
            except Exception as e:
                print("MODEL ERROR", e)
                return
            
            _, predicted = outputs.max(dim=1)
            print(predicted)
            try:
                for fileName, sensorId, pred in zip(fileNames, sensorIds, predicted):
                    name=fileName.split(".")[0]
                    # print(fileName)
                    current_date=datetime.now().strftime("%Y-%m-%d%H-%M-%S-%f")
                    csv_file=open(f"N:\\{sensorId}\\predictions_{name}_{current_date}.csv'", mode='w', newline='')
                    writer = csv.writer(csv_file)
                    writer.writerow(['Timestamp','FileName','SensorId', 'Prediction'])
                    writer.writerow([datetime.now().isoformat(), fileName, sensorId, pred.item()])
                    csv_file.close()
                    
            except Exception as e:
                print("ERROR WRITING TO FILE", e)
            batchedInput.clear()       
    async def asyncmodelWrapper(decoder,batchedInput,index2,writer):
        loop = asyncio.get_event_loop()
        # print(body)
        await loop.run_in_executor(None, modelWrapper,decoder,batchedInput,index2)                       
                 

    # async def runModelTasks(batchedInput):
    #     nonlocal tasksModel
    #     global sam_decoder
    #     for index,decoderModel in enumerate(sam_decoder):
    #         tasksModel.append(asyncmodelWrapper(decoderModel,batchedInput[:(index+1)*32],index))
    #     await asyncio.gather(*tasksModel)
    # Define a callback function to process the received messages
    @sync
    async def callback(ch, method, properties, body):
        nonlocal batchedInput1
        # nonlocal batchedInput2
        nonlocal counter
        global DEVICE
        global sam_decoder
        # global sam_decoder2
        nonlocal transform
        nonlocal pictureId
        global tasks
        global loop
        nonlocal tasksModel
        nonlocal flagModel
        nonlocal valueDivider
        global countModel
        # print(sam_decoder)
        # print(properties.headers['counter'])
        # print("ddddwekdld")
        task=asyncio.create_task(helpers.DrawImage(body,pictureId))  
        # print(task,"SSSASas")
        tasks.append(task)
        # print(tasks)
        # print("Sasjaadjhdad, task appended")
        object={'image':"./file_"+str(pictureId)+".jpg",'original_size':(264,264),'sensorId':properties.headers['sensorId'],'fileName':properties.headers['file_name']}
        pictureId=pictureId+1
        counter=counter+1
        # print("sdnjsdkjsjd, counter")
        # if(flagModel==False):
        batchedInput1.append(object)
        # else:
            # batchedInput2.append(object)
        if(counter%valueDivider==0 ):
            # print(tasks)     
            # print(sam_decoder2)
            # await asyncio.run(run_tasks())
            # print(len(tasks))
            tasksToRun=tasks[:valueDivider]
            tasks=tasks[valueDivider:]
            await asyncio.gather(*tasksToRun)
            for index, decoderModel in enumerate(sam_decoder):
                tasksModel.append(asyncio.create_task(modelWrapper(decoderModel, batchedInput1[:(index + 1) * 32], index)))

            counter=0
            pictureId=0
            # batchedInput1=[]
            # tasks=tasks[valueDivider:]
                    # flagModel=True
                    # tasksModel.append(asyncio.run_coroutine_threadsafe(asyncmodelWrapper(sam_decoder2,batchedInput1,flagModel),loop))
                    # flagModel=False
                    # pictureId=0
        if(len(tasksModel)%countModel==0 and len(tasksModel)!=0):
            print(tasksModel)
            # threading.Lock()
            tasksToRunModel=tasksModel[:countModel]
            tasksModel=tasksModel[countModel:]
            await asyncio.gather(*tasksToRunModel)
            print("dshsfsjf, modle closed")
           
            #  print(batchedInput)
           

        # Save the file content
        # with open(output_file_path, 'wb') as file:
        #     file.write(body)
        # print(f" [x] Received and saved file to '{output_file_path}'")
        # print(f" [x] Received headers: {properties.headers}")


        # Acknowledge the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    # Consume messages from the queue
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

# Example usage
# import threading
# consumer_thread = threading.Thread(target=start_background_loop,args=loop,daemon=True)
# consumer_thread.start()

# Run the asyncio event loop
if __name__ == '__main__':
    try:
        # receive_file()
        receive_file()
        web.run_app(app)
    except NameError as error:
        print("errro",error)
        pass
