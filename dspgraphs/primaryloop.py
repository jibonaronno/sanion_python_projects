import threading
import tkinter as tk
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
    def __init__(self, rootParent:tk.Tk, serialPort):
        self.serialport = serialPort
        self.stop_event = threading.Event()
        self.root = rootParent
        super().__init__()

    def run(self):
        dripA = 0
        dripB = 0
        in_waiting = 0
        jMessage = ""
        unit = ''
        itm = ''
        while not self.stop_event.is_set():
            try:
                in_waiting = self.serialport.in_waiting
            except Exception as e:
                print('Ex:0X07 : ' + str(e))

            while in_waiting == 0:
                # For test only. Commented out
                # dripB = dripB + 1
                # if dripB > 200:
                #     dripB = 0
                #     # self.root.event_generate("<<DataAvailable>>", when="tail", data="TEST DATA")
                ###  time.sleep(0.0006)
                # if dripA > 0:
                #     dripA = dripA + 1
                #     if dripA > 1:
                #         self.root.event_generate("<<DataAvailable>>", when="tail", data=itm)
                #         #  time.sleep(0.01)
                #         itm = ""
                #         dripA = 0
                try:
                    #  print("self.serialport.in_waiting")
                    in_waiting = self.serialport.in_waiting
                    if self.stop_event.is_set():
                        break
                except Exception as e:
                    print('Ex:0x08 : ' + str(e))

            #### time.sleep(0.01)
            try:
                ##  time.sleep(0.001)
                in_waiting = self.serialport.in_waiting
                unit = self.serialport.read(1)
                dripA = 1
                ## unit = self.serialport.readline().decode('utf-8').rstrip()
                ###  unit = self.serialport.readline() #  .rstrip()
                #### print(unit)
                #### print(str(in_waiting))
                ##self.serialport.flushInput()
            except Exception as e:
                print('Ex in sensor Thread readline() 52 : ' + str(e))

            if len(unit) > 0:
                try:
                    ##  itm = unit.decode('Ascii')
                    if unit == b'\r':
                        pass
                    elif unit != b'\n':
                        itm = itm + unit.decode('utf-8')
                    else:
                        itm = itm + unit.decode('utf-8')
                        self.root.event_generate("<<DataAvailable>>", when="tail", data=itm)
                        itm = ''
                except Exception as e:
                    print(f"Error In unit.decode(ascii) {str(e)}")

        self.serialport.close()
        print("Stop Event Is Set")
    def stop(self):
        self.stop_event.set()