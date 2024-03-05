
#  DASH class self.com_ports is a list of available ports
#  DASH class self.comportlisttree is the list box of com ports at left
#  DASH class self.check_variable_lf check box boolean variable for Line Feed

#  Following [Creating Thread Object] section creates the serial data read / write threading
#  class object self.sensor_thread . I want to send data from thread to TkInter GUI by event handler.
#  It is required to call a specific TCL function to bind callback to receive user string data from
#  the threading.Thread derived class. Solution came from
#  https://stackoverflow.com/questions/41912004/how-to-use-tcl-tk-bind-function-on-tkinters-widgets-in-python

#  SensorThread Class derived from threading.Thread written in primaryloop.py file.
#  Class is written similar style like QtThread. But here we use pythons threading.Thread
#  event_generate method of tk is used to send event with data to the main tk window from inside
#  the thread. Serial data send event is called "DataAvailable" . Events are created with the
#  same name from sender class to receiver. Here the receiver is the main window.

#  Check that serial reading thread is started in connect_to_port(self): function for now.
#  or we can shift it to under the button btnStartCollect

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

import serial
from serial.tools import list_ports
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from figures import FIGS
from primaryloop import SensorThread
from threading import Thread, Event
import queue
import time



class DASH(object):
    def __init__(self):
        #  self.parent = _parent
        self.root = tk.Tk()
        self.root.title("DASH")
        self.root.state("zoomed")
        self.side_frame = tk.Frame(self.root, borderwidth=1, relief="groove")  #,yscrollcommand=scrollbar.set)
        self.side_frame.pack(side="left") #  , fill="y")

        self.frameX = tk.Frame(self.root, borderwidth=1, relief="groove")
        self.frameX.pack(side="top")

        self.label = tk.Label(self.side_frame, text="Dashboard", bg="#4C2A85", fg="#FFF", font=25)
        self.label.pack(side="top", pady=10, padx=5)

        self.frameA = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameA.pack() #  fill='y')

        self.btnRead = tk.Button(self.frameA, text="Read Again", command=self.readAgain)
        self.btnConnect = tk.Button(self.frameA, text="Connect", command=self.connect_to_port, state="disabled")
        self.btnRedraw = tk.Button(self.frameA, text="Redraw", command=self.redrawFigs)
        self.btnRead.pack(side="left")
        self.btnConnect.pack(side="left")
        self.btnRedraw.pack(side="left")
        self.charts_frame = tk.Frame(self.root, borderwidth=2, relief="raised")
        self.charts_frame.pack(side="top", fill='y', padx=5, pady=5)
        self.upper_frame = tk.Frame(self.charts_frame)
        self.upper_frame.pack(fill="both", expand=True)
        self.canvases = []

        self.datalist = []
        self.datalist.append([])

        self.comportlisttree = tk.Listbox(self.side_frame, width=50)
        self.comportlisttree.bind("<<ListboxSelect>>", self.on_select_list_item)
        self.comportlisttree.pack(padx=5, pady=5)
        self.com_ports = []
        self.list_com_ports()
        self.text_for_tx = ""

        self.frameB = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameB.pack()  # fill='y')
        self.textbox_tx = tk.Text(self.frameB, height=2, width=30) #  , width=40)
        self.textbox_tx.pack(side="left", padx=5, pady=5)
        self.check_variable_lf = tk.BooleanVar()
        self.checkbox_lf = tk.Checkbutton(self.frameB, text="LF", variable=self.check_variable_lf)
        self.checkbox_lf.pack(side="bottom")
        self.figs = FIGS()
        #  self.redrawFigs()

        self.frameC = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameC.pack() #  fill='y')
        self.btnStartCollect = tk.Button(self.frameC, text="Start Collect", command=self.startCollect)
        self.btnStartCollect.pack(side=tk.LEFT)
        self.btnSend = tk.Button(self.frameC, text="Send", command=self.sendSerial)
        self.btnSend.pack(side=tk.LEFT)

        self.frameD = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameD.pack() #  fill='y')
        self.serialdatalistbox = tk.Listbox(self.frameD, width=50)
        self.serialdatalistbox.pack(padx=5, pady=5)

        self.ser = None

        self.serial_port = ""
        self.sensor_thread = None # SensorThread()
        self.serial_queue = queue.Queue()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Ensure clean exit

    def sendSerial(self):
        txt = self.textbox_tx.get("1.0", "end-1c")
        print(txt)
        if self.ser:
            if self.ser.is_open:
                self.ser.write(str(txt).encode("utf-8"))
                self.datalist[0].clear()


    def startCollect(self):
        #  Starting Thread
        self.sensor_thread.start()

    def on_closing(self):
        if self.sensor_thread:
            self.sensor_thread.stop()
            self.sensor_thread.stop_event.set()
        #self.sensor_thread.join()
        self.root.destroy()

    def redrawFigs(self):
        self.figs.addSampleCanvas(self.charts_frame)
        #self.figs.updatePlot([1], self.datalist[0])

    def show(self):
        self.root.mainloop()

    def list_com_ports(self):
        self.comportlisttree.delete(0, tk.END)
        self.com_ports.clear()
        #  com_ports = [port.device for port in list_ports.comports()]
        com_ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(com_ports):
            self.comportlisttree.insert(tk.END, f"{port}: {desc} [{hwid}]")
            self.com_ports.append(port)

    #  Event when com port list item is selected
    def on_select_list_item(self, event):
        selected_index = self.comportlisttree.curselection()
        if selected_index:
            selected_port = self.comportlisttree.get(selected_index)
            print("Selected Port : ", selected_port)
            self.btnConnect.config(state="normal")

    def onSerialDataReceived(self, event):
        if event:
            # data = event.__getattribute__("data")
            # print(event.__dict__)
            self.serialdatalistbox.insert(self.serialdatalistbox.size(), "%s" % event)
            dta = "%s" % event
            self.datalist[0].append(int(str(dta)))
            #  self.serialdatalistbox.insert(self.serialdatalistbox.size(), "Event")
    def connect_to_port(self):
        selected_index = self.comportlisttree.curselection()
        #  print(selected_index) #  selected_index is a tuple like (0, )
        if selected_index:
            selected_port = self.com_ports[selected_index[0]] #  self.comportlisttree.get(selected_index)
            self.serial_port = selected_port
            try:
                ser = serial.Serial(selected_port, baudrate=15200, timeout=0)
                #  timeout parameter in seconds for reading timeout. It is optional. If it is not given or
                #  None, read operation will block the execution.
                print("COM PORT Connected")
                self.ser = ser
                #  Creating Thread Object
                self.sensor_thread = SensorThread(rootParent=self.root, serialPort=ser)
                #  This function did not work : self.root.bind("<<DataAvailable>>", self.onSerialDataReceived)
                #  I was trying to send the serial data from primaryloop.py file to the main.py root parents listbox.
                #  You need to call TCL function to bind the callback to receive the data member which is %d .
                #  Following solution came from
                #  https://stackoverflow.com/questions/41912004/how-to-use-tcl-tk-bind-function-on-tkinters-widgets-in-python
                cmd = self.root.register(self.onSerialDataReceived)
                self.root.tk.call("bind", self.root, "<<DataAvailable>>", cmd + " %d")

            except serial.SerialException as e:
                print("Error Serial Port Connection", e)

    #  Write a command to the COM port (example: sending 'Hello')
    #  ser.write(b'Hello\n')  # The device connected to the COM port needs to understand this command

    def readAgain(self):
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dash = DASH()
    dash.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
