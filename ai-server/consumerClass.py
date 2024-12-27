import functools
import logging
import time
import pika
from pika.exchange_type import ExchangeType
import asyncio
import numpy as np
import functools
from SamDecoder import SAM_Decoder
from modeling import ImageEncoderViT
from torch.nn import functional as F
from torchvision.io import read_image
from torchvision import transforms,datasets
from PIL import Image
from torch.utils.data import DataLoader,Dataset
import os
import torch
import helpers as helpers
from datetime import datetime
import csv
import aiormq
from pika.adapters.asyncio_connection import AsyncioConnection
from aiohttp import web
import threading
import socketio
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)
sid=None

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
Logger = logging.getLogger(__name__)
# loop=asyncio.new_event_loop()


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
            raise ("image not found ")
        # print(image)
        if self.transform:
            image = self.transform(image)
        return image  # Returning the image and its path


class Consumer(object):
    """This is an example consumer that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, this class will stop and indicate
    that reconnection is necessary. You should look at the output, as
    there are limited reasons why the connection may be closed, which
    usually are tied to permission related issues or socket timeouts.

    If the channel is closed, it will indicate a problem with one of the
    commands that were issued and that should surface in the output as well.

    """
    EXCHANGE = 'file_exchange'
    EXCHANGE_TYPE = ExchangeType.topic
    QUEUE = 'file_queue'
    ROUTING_KEY = "file_queue"

    def __init__(self, amqp_url):
        """Create a new instance of the consumer class, passing in the AMQP
        URL used to connect to RabbitMQ.

        :param str amqp_url: The AMQP url to connect with

        """
        self.should_reconnect = False
        self.was_consuming = False

        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = amqp_url
        self._consuming = False
        self.batchedInput1=[]
        self.counter=0
        self.pictureId=0
        self.tasksModel=[]
        self.tasks=[]
        self.countModel=2
        self.valueDivider=32* self.countModel
        self.transform = transforms.Compose([
            transforms.Resize((264, 264)),
            transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.sam_decoder=[]
        self.DEVICE=torch.device('cuda:0')

        # In production, experiment with higher prefetch values
        # for higher consumer throughput
        self._prefetch_count = 1
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        self.PATH= './sam_classification_model_'+ str(7) +'_6.pth'
        self.keysMapping={0:'gis_corona', 1:'gis_floating' , 2:'gis_particle', 3:'gis_void', 4:'mtr_corona', 5:'mtr_floating', 6: 'mtr_particle', 7: 'mtr_surface', 8: 'mtr_void'}
        torch.cuda.empty_cache() 
        self.load_model()
        os.makedirs("N:\\SPDC_Data_Diagonsis",exist_ok=True)
        os.makedirs("N:\\SPDC_Data_Diagonsis\\GLU",exist_ok=True)
        os.makedirs("N:\\SPDC_Data_Diagonsis\\MLU",exist_ok=True)


    def load_model(self):
        for x in range(self.countModel):
            sam_decoder2=SAM_Decoder(sam_encoder=ImageEncoderViT)
            sam_decoder2.load_state_dict(torch.load('./sam_classification_model_7_6.pth',weights_only=True))
            sam_decoder2=sam_decoder2.to(self.DEVICE)
            # sam_decoder.eval()
            sam_decoder2.eval()
            self.sam_decoder.append(sam_decoder2)
        print(self.sam_decoder)

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        Logger.info('Connecting to %s', self._url)
        return AsyncioConnection(
            parameters=pika.URLParameters(self._url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed
        )


    def close_connection(self):
        self._consuming = False
        if self._connection.is_closing or self._connection.is_closed:
            print('Connection is closing or already closed')
        else:
            Logger.info('Closing connection')
            self._connection.close()

    def on_connection_open(self, _unused_connection):
        """This method is called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :param pika.SelectConnection _unused_connection: The connection

        """
        Logger.info('Connection opened')
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        """This method is called by pika if the connection to RabbitMQ
        can't be established.

        :param pika.SelectConnection _unused_connection: The connection
        :param Exception err: The error

        """
        Logger.error('Connection open failed: %s', err)
        self.reconnect()

    def on_connection_closed(self, _unused_connection, reason):
        """This method is invoked by pika when the connection to RabbitMQ is
        closed unexpectedly. Since it is unexpected, we will reconnect to
        RabbitMQ if it disconnects.

        :param pika.connection.Connection connection: The closed connection obj
        :param Exception reason: exception representing reason for loss of
            connection.

        """
        self._channel = None
        if self._closing:
            self._connection.ioloop.stop()
        else:
            Logger.warning('Connection closed, reconnect necessary: %s', reason)
            self.reconnect()

    def reconnect(self):
        """Will be invoked if the connection can't be opened or is
        closed. Indicates that a reconnect is necessary then stops the
        ioloop.

        """
        self.should_reconnect = True
        self.stop()

    def open_channel(self):
        """Open a new channel with RabbitMQ by issuing the Channel.Open RPC
        command. When RabbitMQ responds that the channel is open, the
        on_channel_open callback will be invoked by pika.

        """
        Logger.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        """This method is invoked by pika when the channel has been opened.
        The channel object is passed in so we can make use of it.

        Since the channel is now open, we'll declare the exchange to use.

        :param pika.channel.Channel channel: The channel object

        """
        Logger.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        """This method tells pika to call the on_channel_closed method if
        RabbitMQ unexpectedly closes the channel.

        """
        Logger.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        """Invoked by pika when RabbitMQ unexpectedly closes the channel.
        Channels are usually closed if you attempt to do something that
        violates the protocol, such as re-declare an exchange or queue with
        different parameters. In this case, we'll close the connection
        to shutdown the object.

        :param pika.channel.Channel: The closed channel
        :param Exception reason: why the channel was closed

        """
        Logger.warning('Channel %i was closed: %s', channel, reason)
        self.close_connection()

    def setup_exchange(self, exchange_name):
        """Setup the exchange on RabbitMQ by invoking the Exchange.Declare RPC
        command. When it is complete, the on_exchange_declareok method will
        be invoked by pika.

        :param str|unicode exchange_name: The name of the exchange to declare

        """
        Logger.info('Declaring exchange: %s', exchange_name)
        # Note: using functools.partial is not required, it is demonstrating
        # how arbitrary data can be passed to the callback when it is called
        cb = functools.partial(
            self.on_exchange_declareok, userdata=exchange_name)
        self._channel.exchange_declare(
            exchange=exchange_name,
            callback=cb)
    @staticmethod
    def sync(f):
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if not self._connection.ioloop.is_running():
                return self._connection.ioloop.run_until_complete(f(self, *args, **kwargs))
            else:
                return asyncio.ensure_future(f(self, *args, **kwargs))
        return wrapper

    def on_exchange_declareok(self, _unused_frame, userdata):
        """Invoked by pika when RabbitMQ has finished the Exchange.Declare RPC
        command.

        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response frame
        :param str|unicode userdata: Extra user data (exchange name)

        """
        Logger.info('Exchange declared: %s', userdata)
        self.setup_queue(self.QUEUE)

    def setup_queue(self, queue_name):
        """Setup the queue on RabbitMQ by invoking the Queue.Declare RPC
        command. When it is complete, the on_queue_declareok method will
        be invoked by pika.

        :param str|unicode queue_name: The name of the queue to declare.

        """
        Logger.info('Declaring queue %s', queue_name)
        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        self._channel.queue_declare(queue=queue_name, callback=cb)

    def on_queue_declareok(self, _unused_frame, userdata):
        """Method invoked by pika when the Queue.Declare RPC call made in
        setup_queue has completed. In this method we will bind the queue
        and exchange together with the routing key by issuing the Queue.Bind
        RPC command. When this command is complete, the on_bindok method will
        be invoked by pika.

        :param pika.frame.Method _unused_frame: The Queue.DeclareOk frame
        :param str|unicode userdata: Extra user data (queue name)

        """
        queue_name = userdata
        Logger.info('Binding %s to %s with %s', self.EXCHANGE, queue_name,
                   self.ROUTING_KEY)
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        self._channel.queue_bind(
            queue_name,
            self.EXCHANGE,
            routing_key=self.ROUTING_KEY,
            callback=cb)

    def on_bindok(self, _unused_frame, userdata):
        """Invoked by pika when the Queue.Bind method has completed. At this
        point we will set the prefetch count for the channel.

        :param pika.frame.Method _unused_frame: The Queue.BindOk response frame
        :param str|unicode userdata: Extra user data (queue name)

        """
        Logger.info('Queue bound: %s', userdata)
        self.set_qos()

    def set_qos(self):
        """This method sets up the consumer prefetch to only be delivered
        one message at a time. The consumer must acknowledge this message
        before RabbitMQ will deliver another one. You should experiment
        with different prefetch values to achieve desired performance.

        """
        self._channel.basic_qos(
            prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok)

    def on_basic_qos_ok(self, _unused_frame):
        """Invoked by pika when the Basic.QoS method has completed. At this
        point we will start consuming messages by calling start_consuming
        which will invoke the needed RPC commands to start the process.

        :param pika.frame.Method _unused_frame: The Basic.QosOk response frame

        """
        Logger.info('QOS set to: %d', self._prefetch_count)
    # Declare a queue (if it doesn't already exist)
    
        self.start_consuming()

    def start_consuming(self):
        """This method sets up the consumer by first calling
        add_on_cancel_callback so that the object is notified if RabbitMQ
        cancels the consumer. It then issues the Basic.Consume RPC command
        which returns the consumer tag that is used to uniquely identify the
        consumer with RabbitMQ. We keep the value to use it when we want to
        cancel consuming. The on_message method is passed in as a callback pika
        will invoke when a message is fully received.

        """
        Logger.info('Issuing consumer related RPC commands')
        self.add_on_cancel_callback()
        self._consumer_tag = self._channel.basic_consume(
            self.QUEUE, self.on_callback)
        self.was_consuming = True
        self._consuming = True

    def add_on_cancel_callback(self):
        """Add a callback that will be invoked if RabbitMQ cancels the consumer
        for some reason. If RabbitMQ does cancel the consumer,
        on_consumer_cancelled will be invoked by pika.

        """
        Logger.info('Adding consumer cancellation callback')
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        """Invoked by pika when RabbitMQ sends a Basic.Cancel for a consumer
        receiving messages.

        :param pika.frame.Method method_frame: The Basic.Cancel frame

        """
        Logger.info('Consumer was cancelled remotely, shutting down: %r',
                  method_frame)
        self._channel.close()

    # def on_message(self, _unused_channel, basic_deliver, properties, body):
    #     """Invoked by pika when a message is delivered from RabbitMQ. The
    #     channel is passed for your convenience. The basic_deliver object that
    #     is passed in carries the exchange, routing key, delivery tag and
    #     a redelivered flag for the message. The properties passed in is an
    #     instance of BasicProperties with the message properties and the body
    #     is the message that was sent.

    #     :param pika.channel.Channel _unused_channel: The channel object
    #     :param pika.Spec.Basic.Deliver: basic_deliver method
    #     :param pika.Spec.BasicProperties: properties
    #     :param bytes body: The message body

    #     """
    #     # Logger.info('Received message # %s from %s: %s',
    #                 basic_deliver.delivery_tag, properties.app_id, body)
    #     self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        """Acknowledge the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.

        :param int delivery_tag: The delivery tag from the Basic.Deliver frame

        """
        Logger.info('Acknowledging message %s', delivery_tag)
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        """Tell RabbitMQ that you would like to stop consuming by sending the
        Basic.Cancel RPC command.

        """
        if self._channel:
            Logger.info('Sending a Basic.Cancel RPC command to RabbitMQ')
            cb = functools.partial(
                self.on_cancelok, userdata=self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _unused_frame, userdata):
        """This method is invoked by pika when RabbitMQ acknowledges the
        cancellation of a consumer. At this point we will close the channel.
        This will invoke the on_channel_closed method once the channel has been
        closed, which will in-turn close the connection.

        :param pika.frame.Method _unused_frame: The Basic.CancelOk frame
        :param str|unicode userdata: Extra user data (consumer tag)

        """
        self._consuming = False
        Logger.info(
           'RabbitMQ acknowledged the cancellation of the consumer: %s',
           userdata)
        self.close_channel()

    def close_channel(self):
        """Call to close the channel with RabbitMQ cleanly by issuing the
        Channel.Close RPC command.

        """
        Logger.info('Closing the channel')
        self._channel.close()


    def run(self):
        """Run the example consumer by connecting to RabbitMQ and then
        starting the IOLoop to block and allow the AsyncioConnection to operate.

        """
        self._connection = self.connect()
        self._connection.ioloop.run_forever()

    def stop(self):
        """Cleanly shutdown the connection to RabbitMQ by stopping the consumer
        with RabbitMQ. When RabbitMQ confirms the cancellation, on_cancelok
        will be invoked by pika, which will then closing the channel and
        connection. The IOLoop is started again because this method is invoked
        when CTRL-C is pressed raising a KeyboardInterrupt exception. This
        exception stops the IOLoop which needs to be running for pika to
        communicate with RabbitMQ. All of the commands issued prior to starting
        the IOLoop will be buffered but not processed.

        """
        if not self._closing:
            self._closing = True
            Logger.info('Stopping')
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.run_forever()
            else:
                self._connection.ioloop.stop()
            Logger.info('Stopped')

    @sync
    async def on_callback(self,ch,method, properties, body):
        # print(sam_decoder)
        # print(properties.headers['counter'])
        # print("ddddwekdld")
        # # Logger.info('Received message # %s from %s: %s',
        #             method.delivery_tag, properties.app_id, body)
        task=asyncio.create_task(helpers.DrawImage(body,self.pictureId))  
        # print(task,"SSSASas")
        self.tasks.append(task)
        # print(tasks)
        # print("Sasjaadjhdad, task appended")
        dictionaryLUNames={}
        dictionaryLNNames={}
        path="N:\\SPDC_Data_Diagonsis\\"
        if properties.headers["GLUId"]!=0:
            path=path+"GLU\\"
            if properties.headers["LUName"] in dictionaryLUNames and dictionaryLUNames[properties.headers["LUName"]]!=0:
                if properties.headers['LNName'] in dictionaryLNNames and dictionaryLNNames[properties.headers['LNName']]!=0:
                    path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"
                else:
                    os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"])
                    path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"
            else:
                os.makedirs(path+properties.headers["LUName"],exist_ok=True)
                os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"],exist_ok=True)
                os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime",exist_ok=True)
                path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"
        if properties.headers["MLUId"]!=0:
            path=path+"MLU\\"
            if properties.headers["LUName"] in dictionaryLUNames and dictionaryLUNames[properties.headers["LUName"]]!=0:
                if properties.headers['LNName'] in dictionaryLNNames and dictionaryLNNames[properties.headers['LNName']]!=0:
                    path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"
                else:
                    os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"])
                    path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"
            else:
                os.makedirs(path+properties.headers["LUName"],exist_ok=True)
                os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"],exist_ok=True)
                os.makedirs(path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime",exist_ok=True)
                path=path+properties.headers["LUName"]+"\\"+ properties.headers["LNName"]+"\\RealTime"

        object={'image':"./file_"+str(self.pictureId)+".jpg",'original_size':(264,264),'SensorInfoId':properties.headers['SensorInfoId'],'FullName':properties.headers['FullName'],'EventId':properties.headers["EventId"],'LNName':properties.headers["LNName"],
                'LUName':properties.headers["LUName"],'GLUId':properties.headers["GLUId"],'MLUId':properties.headers["MLUId"],"path":path}
        # print(object)
        # print(path)
        self.pictureId=self.pictureId+1
        self.counter=self.counter+1
        # print("sdnjsdkjsjd, counter")
        # if(flagModel==False):
        self.batchedInput1.append(object)
        # print("batch len", len(self.batchedInput1))
        # else:
            # batchedInput2.append(object)
        if(self.counter%self.valueDivider==0 ):
            # print(tasks)     
            # print(sam_decoder2)
            # await asyncio.run(run_tasks())
            # print(len(tasks))
            tasksToRun=self.tasks[:self.valueDivider]
            self.tasks=self.tasks[self.valueDivider:]
            await asyncio.gather(*tasksToRun)
            for index, decoderModel in enumerate(self.sam_decoder):
                print('decoder index', index)
                print(len(self.batchedInput1))
                if index==0:
                    self.tasksModel.append(asyncio.create_task(self.modelWrapper(decoderModel, self.batchedInput1[:(index + 1) * 32], index,path)))
                else: 
                    self.tasksModel.append(asyncio.create_task(self.modelWrapper(decoderModel, self.batchedInput1[32:], index,path)))


            self.counter=0
            self.pictureId=0
            # batchedInput1=[]
            # tasks=tasks[valueDivider:]
                    # flagModel=True
                    # tasksModel.append(asyncio.run_coroutine_threadsafe(asyncmodelWrapper(sam_decoder2,batchedInput1,flagModel),loop))
                    # flagModel=False
                    # pictureId=0
        if(len(self.tasksModel)%self.countModel==0 and len(self.tasksModel)!=0):
            print(self.tasksModel)
            # threading.Lock()
            tasksToRunModel=self.tasksModel[:self.countModel]
            self.tasksModel=self.tasksModel[self.countModel:]
            await asyncio.gather(*tasksToRunModel)
            self.batchedInput1.clear()       

            print("model closed")

        self.acknowledge_message(method.delivery_tag)


    async def modelWrapper(self,decoder, batchedInput, index2,path):
        global sid
        with torch.no_grad():
            multplier = index2 * 32
            print(multplier)
            # print([(f"R:\\file_{multplier + i}.jpg") for i in range(32)])
            input_images = torch.stack([self.transform(Image.open(f"R:\\file_{multplier + i}.jpg")) for i in range(32)], dim=0).to(self.DEVICE)

            # print(batchedInput)
            sensorInfoIds=[]
            fullnames=[]
            eventIds=[]
            lnNames=[]
            luNames=[]
            gluIds=[]
            mluIds=[]
            paths=[]
            for x in batchedInput:
                sensorInfoIds.append(x['SensorInfoId'])
                fullnames.append( x['FullName']) 
                eventIds.append(x['EventId'])
                lnNames.append(x['LNName'])
                luNames.append(x['LUName'])
                gluIds.append(x['GLUId'])
                mluIds.append(x['MLUId'])
                paths.append(x["path"])

            # print(batchedInput)


            try:
                outputs = decoder(input_images)
            except Exception as e:
                print("MODEL ERROR", e)
                return
            
            # _, predicted = outputs.max(dim=1)
            # print(predicted)
            try:
                
                for fullname, sensorInfoId, pred,eventId,lnName,luName,gluId,mluId,path2 in zip(fullnames, sensorInfoIds, outputs,eventIds,lnNames,luNames,gluIds,mluIds,paths):
                    name=fullname
                    # print(fileName)
                    current_date=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
                    x=pred.cpu().tolist()
                    # print("event id", eventId)
                    # csv_file=open(f"N:\\{sensorId}\\predictions_{name}_{current_date}.csv", mode='w', newline='')
                    with open(f"{path2}\\predictions_{name}_{current_date}.csv", mode='w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(['Timestamp', 'FullName', 'SensorInfoId','EventId','GLUId','MLUId','LUName','LNName','Classes','Prediction'])
                        writer.writerow([datetime.now().isoformat(), fullname,sensorInfoId,eventId,gluId,mluId,luName,lnName,self.keysMapping,x ])
                    
                    if(eventId!=0):
                        print('sending socket!!!', eventId)
                        data={
                            'Timestamp':datetime.now().isoformat(), 'FullName':fullname, 'SensorInfoId':sensorInfoId,
                            'EventId':eventId,'GLUId':gluId,'MLUId':mluId,'LUName':luName,'LNName':lnName,
                            'Classes':self.keysMapping,'Prediction':x
                        }
                        await chat_message(sid,data,self._connection.ioloop)   
            except Exception as e:
                print("ERROR WRITING TO FILE", e)
            batchedInput.clear()
 
    

class ReconnectingConsumer(object):
    """This is an example consumer that will reconnect if the nested
    ExampleConsumer indicates that a reconnect is necessary.

    """

    def __init__(self, amqp_url):
        self._reconnect_delay = 0
        self._amqp_url = amqp_url
        self._consumer = Consumer(self._amqp_url)

    def run(self):
        while True:
            try:
                self._consumer.run()
            except KeyboardInterrupt:
                self._consumer.stop()
                break
            self._maybe_reconnect()

    def _maybe_reconnect(self):
        if self._consumer.should_reconnect:
            self._consumer.stop()
            reconnect_delay = self._get_reconnect_delay()
            Logger.info('Reconnecting after %d seconds', reconnect_delay)
            time.sleep(reconnect_delay)
            self._consumer = Consumer(self._amqp_url)

    def _get_reconnect_delay(self):
        if self._consumer.was_consuming:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay
    
def create_ram_drive(size_mb, drive_letter):
    # Ensure imdisk is installed and available in PATH
    os.system(f'imdisk -a -t vm -s {size_mb}M -m {drive_letter}: -p "/fs:NTFS /q /y"')

# Example usage



@sio.event
async def connect(sid, environ):
    sid=sid
    print("connected")
    await sio.emit("my response", {"data": "Connected"})

@sio.event
async def chat_message(sid, data,loop):
   print("diagonsis",data)
   await sio.emit('diagnosisEv', {'data': data})

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

async def handle(request):
    print("Abc")

app.add_routes([
    web.get('/', handle)])

async def main(stop_event):
    # Example of setting up your RAM drive (assuming you have this function defined)
    # create_ram_drive(1024, 'R')
    # logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
    create_ram_drive(1024, 'R')

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner)

    loop = asyncio.get_event_loop()
    asyncio.run_coroutine_threadsafe(site.start(), loop)

    amqp_url = 'amqp://guest:guest@127.0.0.1:5672/%2F'
    consumer = ReconnectingConsumer(amqp_url)
    asyncio.run_coroutine_threadsafe(consumer.run(), loop)
    # Wait until the stop_event is set
    await stop_event.wait()

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    # Run the event loop in a separate thread
    def run_loop():
        loop.run_forever()

    loop_thread = threading.Thread(target=run_loop)
    loop_thread.start()

    # Schedule the main coroutine to run on the event loop
    asyncio.run_coroutine_threadsafe(main(stop_event), loop)

    try:
        # Keep the main thread alive while the event loop is running
        while loop_thread.is_alive():
            loop_thread.join(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        loop.call_soon_threadsafe(stop_event.set)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join()