# #import pika, os
# import asyncio,time

# # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
# from rstream import Producer, AMQPMessage

# # Create a producer

# # Create a Stream named 'mystream'

# # Construct the message



# async def main():
#     producer = Producer(
#     host='localhost',
#     port=5552,
#     username='guest',
#     password='guest',
#   )
#     message = AMQPMessage(
#       body=bytes('hello world', "utf-8")
#     )
#     # Assuming consumer is an instance of Consumer
#     await producer.create_stream('mystream', exists_ok=True)
#     start_time = time.perf_counter()

#     for i in range(10000):
#       amqp_message = AMQPMessage(
#           body=bytes("hello: {}".format(i), "utf-8"))
#         # send is asynchronous
#       print(message)
#       await producer.send(stream='mystream', message=amqp_message)

#     end_time = time.perf_counter()
# # Python 3.7+
# asyncio.run(main())

# Publish the message
import pika
import sys
# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host='localhost'))
# channel = connection.channel()

# channel.exchange_declare(exchange='logs', exchange_type='fanout')

# message = ' '.join(sys.argv[1:]) or "info: Hello World!"
# channel.basic_publish(exchange='logs', routing_key='', body=message)
# print(f" [x] Sent {message}")
# connection.close()


def send_file(file_path, queue_name):
    # Establish a connection to RabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Declare a queue (if it doesn't already exist)
    channel.queue_declare(queue=queue_name)

    # Read the file content
    with open(file_path, 'rb') as file:
        file_content = file.read()
    headers={
        'sensorId':1,
    }
    properties = pika.BasicProperties(headers=headers)
    # Publish the file content to the queue
    channel.basic_publish(exchange='',
                          routing_key=queue_name,
                          body=file_content,
                          properties=properties)
    print(f" [x] Sent file '{file_path}'")

    # Close the connection
    connection.close()

# Example usage
send_file('./abc.dat', 'myQueue')