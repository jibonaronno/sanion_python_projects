
#  DASH class self.com_ports is a list of available ports
#  DASH class self.comportlisttree is the list box of com ports at left
#  DASH class self.check_variable_lf check box boolean variable for Line Feed

#  Following [Creating Thread Object] section creates the serial data read / write threading
#  class object self.sensor_thread . I want to send data from thread to TkInter GUI by event handler.
#

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
        self.side_frame.pack(side="left", fill="y")

        self.frameX = tk.Frame(self.root, borderwidth=1, relief="groove")
        self.frameX.pack(side="top")

        self.label = tk.Label(self.side_frame, text="Dashboard", bg="#4C2A85", fg="#FFF", font=25)
        self.label.pack(side="top", pady=50, padx=20)

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

        self.comportlisttree = tk.Listbox(self.side_frame, width=50)
        self.comportlisttree.bind("<<ListboxSelect>>", self.on_select_list_item)
        self.comportlisttree.pack(padx=5, pady=5)
        self.com_ports = []
        self.list_com_ports()
        self.text_for_tx = ""

        self.frameB = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameB.pack(fill='y')  # fill='y')
        self.textbox_tx = tk.Text(self.frameB, height=2) #  , width=40)
        self.textbox_tx.pack(side="left", padx=5, pady=5)
        self.check_variable_lf = tk.BooleanVar()
        self.checkbox_lf = tk.Checkbutton(self.frameB, text="LF", variable=self.check_variable_lf)
        self.checkbox_lf.pack(side="bottom")
        self.figs = FIGS()
        #  self.redrawFigs()

        self.frameC = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameC.pack(fill='y')
        self.btnStartCollect = tk.Button(self.frameC, text="Start Collect", command=self.startCollect)
        self.btnStartCollect.pack(side=tk.LEFT)

        self.serial_port = ""
        self.sensor_thread = None # SensorThread()
        self.serial_queue = queue.Queue()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Ensure clean exit

    def startCollect(self):
        #  Starting Thread
        self.sensor_thread.start()

    def on_closing(self):
        self.sensor_thread.stop()
        self.sensor_thread.stop_event.set()
        #self.sensor_thread.join()
        self.root.destroy()

    def redrawFigs(self):
        self.figs.addSampleCanvas(self.charts_frame)

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

                #  Creating Thread Object
                self.sensor_thread = SensorThread(rootParent=self.root, serialPort=ser)

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
