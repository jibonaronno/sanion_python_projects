import threading

import serial
from threading import Thread, Event
import queue
import time

# def read_serial_data():
#     try:
#         ser = serial.Serial(serial_port, 115200, timeout=1)  # Adjust as needed
#         while not stop_thread.is_set():
#             if ser.in_waiting:
#                 data = ser.readline().decode().strip()
#                 serial_queue.put(data)
#     except serial.SerialException as e:
#         print(f"Serial error: {e}")
#     finally:
#         ser.close()

class SensorThread(threading.Thread):
    def __init__(self, serialPort):
        self.serialport = serialPort
        self.stop_event = threading.Event()
        super().__init__()

    def run(self):
        in_waiting = ''
        jMessage = ""
        unit = ''
        itm = ''
        while not self.stop_event.is_set():
            try:
                in_waiting = self.serialport.in_waiting
            except Exception as e:
                print('Ex:0X07 : ' + str(e))

            while in_waiting == 0:
                time.sleep(0.01)
                try:
                    #  print("self.serialport.in_waiting")
                    in_waiting = self.serialport.in_waiting
                    if self.stop_event.is_set():
                        break
                except Exception as e:
                    print('Ex:0x08 : ' + str(e))

            time.sleep(0.01)
            try:
                print("self.serialport.read(in_waiting)")
                unit = self.serialport.read(in_waiting)
                print("self.serialport.read(in_waiting)")
            except Exception as e:
                print('Ex in sensor Thread readline() 52 : ' + str(e))

            if len(unit) > 0:
                try:
                    itm += unit.decode('ascii')
                except:
                    pass
        self.serialport.close()
        print("Stop Event Is Set")
    def stop(self):
        self.stop_event.set()