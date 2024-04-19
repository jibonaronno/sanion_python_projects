
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

#  Check that self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  this command is used to call
#  on_closing() function when the main windows is about to close.
# Check this read for canvas related all the methods and properties.
# https://tkinter-docs.readthedocs.io/en/latest/widgets/canvas.html
# Here is a canvas example https://python4kids.wordpress.com/2012/09/19/quadratic-bezier-curves/
#

from pprint import pprint
import matplotlib.pyplot as plt
from Bezier import Bezier
import numpy as np
import tkinter as tk
from tkinter import ttk

import serial
from serial.tools import list_ports
from figures import FIGS
from polarblock import POLARBLOCK
from primaryloop import SensorThread
from threading import Thread, Event
import queue
import time
#  from matplotlib.figure import Figure
#  from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DASH(object):
    def __init__(self):
        #  self.parent = _parent
        self.root = tk.Tk()
        self.root.title("DASH")
        #  self.root.state("zoomed")
        self.root.geometry('940x800')
        self.side_frame = tk.Frame(self.root, borderwidth=1, relief="groove")  #,yscrollcommand=scrollbar.set)
        self.side_frame.pack(side="left") #  , fill="y")

        self.frameX = tk.Frame(self.root, borderwidth=1, relief="groove")
        self.frameX.pack(side="top")

        self.label = tk.Label(self.side_frame, text="Dashboard", bg="#4C2A85", fg="#FFF", font=25)
        self.label.pack(side="top", pady=10, padx=5)

        self.frameA = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameA.pack() #  fill='y')

        self.btnConnect = tk.Button(self.frameA, text="Connect", command=self.connect_to_port, state="disabled")
        self.btnRedraw = tk.Button(self.frameA, text="Redraw", command=self.redrawFigs)
        #  Disabled Polar Graph self.btnRedrawPolar = tk.Button(self.frameA, text="Draw Polar", command=self.redrawPolarFigs)

        self.btnConnect.pack(side="left")
        self.btnRedraw.pack(side="left")
        #  Disabled Polar Graph self.btnRedrawPolar.pack(side="left")
        self.btnUpdate = tk.Button(self.frameA, text="Update", command=self.UpdateFigs)
        self.btnUpdate.pack(side=tk.LEFT)
        self.charts_frame = tk.Frame(self.root, borderwidth=2, relief="raised")
        self.charts_frame.pack(side="top", fill='y', padx=5, pady=5)
        self.upper_frame = tk.Frame(self.charts_frame)
        self.upper_frame.pack(fill="both", expand=True)
        self.canvases = []

        self.datalist = []
        self.datalistforgraph = []
        #  self.datalist.append([])
        self.bzr_control_points = np.array([[0, 0], [0, 80], [3, 80], [3, 0], [5.5, 0], [6.5, 0], [7.5, 0], [8.5, 70], [10, 70], [11, 60], [12, 50], [13, 30], [14, 10]])
        self.bzr_total_array = np.arange(0, 1, 0.01)
        self.curve1 = Bezier.Curve(self.bzr_total_array, self.bzr_control_points)


        self.comportlisttree = tk.Listbox(self.side_frame, width=50, height=4)
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
        self.figs = FIGS(self)
        #  self.redrawFigs()

        self.frameC = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameC.pack() #  fill='y')
        self.btnStartCollect = tk.Button(self.frameC, text="Start Collect", command=self.startCollect)
        #  self.btnStartCollect.pack(side=tk.LEFT)
        self.btnSend = tk.Button(self.frameC, text="Send", command=self.sendSerial)
        self.btnSend.pack(side=tk.LEFT)
        self.btnClear = tk.Button(self.frameC, text="Clear", command=self.clearListbox)
        self.btnClear.pack(side=tk.LEFT)
        self.btnRemove = tk.Button(self.frameC, text="Remove", command=self.removeFromList, state="disabled")
        self.btnRemove.pack(side=tk.LEFT)
        self.btnMakeTable = tk.Button(self.frameC, text="Make Table", command=self.start_data_table_timer) # , state="disabled")
        self.btnMakeTable.pack(side=tk.LEFT)

        self.frameD = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameD.pack() #  fill='y')
        #  self.serialdatalistbox = tk.Listbox(self.frameD, width=50)
        #  self.serialdatalistbox.pack(padx=5, pady=5)

        self.frameE = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameE.pack()  # fill='y')
        self.textbox_rx = tk.Text(self.frameE, height=8, width=42)  # , width=40)

        self.textbox_rx_scrollbar = ttk.Scrollbar(self.frameE, orient=tk.VERTICAL, command=self.textbox_rx.yview)
        self.textbox_rx.configure(yscroll=self.textbox_rx_scrollbar.set)
        self.textbox_rx_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.textbox_rx.pack(side="top", padx=5, pady=5)

        self.frameF = tk.Frame(self.side_frame, borderwidth=1, relief="groove")
        self.frameF.pack()  # fill='y')


        self.ser = None
        self.Rtree = None
        self.RtreeScrollbar = None

        self.data_table_redraw_timer_id = None

        self.serial_port = ""
        self.sensor_thread = None # SensorThread()
        self.serial_queue = queue.Queue()
        self.serdata = ""

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Ensure clean exit

    def sendSerial(self):
        txt = self.textbox_tx.get("1.0", "end-1c")
        if self.check_variable_lf.get():
            txt = txt + "\r"
            print("Checked")
        print(txt)
        if self.ser:
            if self.ser.is_open:
                # self.ser.write(str(txt).encode("utf-8"))
                self.ser.write(str(txt).encode("ascii"))
                self.datalist.clear()

    def clearListbox(self):
        #  self.serialdatalistbox.delete(0, tk.END)
        self.datalist.clear()
        self.datalistforgraph.clear()
        if self.Rtree:
            for chld in self.Rtree.get_children():
                self.Rtree.delete(chld)

    def startCollect(self):
        #  Starting Thread
        self.sensor_thread.start()

    def reset_data_table_timer(self):
        if self.data_table_redraw_timer_id:
            self.root.after_cancel(self.data_table_redraw_timer_id)
            self.start_data_table_timer()
        else:
            self.start_data_table_timer()

    def start_data_table_timer(self):
        # Start a timer for 0.5 seconds (500 milliseconds)
        self.data_table_redraw_timer_id = self.root.after(500, self.data_table_timer_callback)

    def data_table_timer_callback(self):
        self.drawTreeTable(len(self.datalist[0]), self.frameF, self.datalist)
        #  self.drawTreeTable(2, self.frameF, self.datalist)
        print(f'Length .datalist {len(self.datalist[0])} , {self.datalist[0]}')

    def drawTreeTable(self, column_count, frame, dta: []):
        if column_count > 0:
            if self.Rtree:
                self.Rtree.destroy()
            if self.RtreeScrollbar:
                self.RtreeScrollbar.destroy()
            columns = []
            column_names = []
            print(str(column_count))
            for idx in range(column_count):
                columns.append('#' + str(idx+1))
                column_names.append("COL:" + str(idx+1))
            self.Rtree = ttk.Treeview(frame, columns=columns, show='headings')
            for col, name in zip(columns, column_names):
                self.Rtree.heading(col, text=name)
            for idx in range(column_count):
                self.Rtree.column(f'#{str(idx+1)}', width=70)
            dtaa = []

            for idx in range(len(dta[0])):
                self.datalistforgraph.append([])

            for unt in dta:
                dtaa.append(unt)
                idxx = 0
                for ele in unt:
                    self.datalistforgraph[idxx].append(ele)
                    idxx = idxx + 1

            print(f"dta : {len(dta)} LENTH: {len(dtaa)}")
            for row in dtaa:
                self.Rtree.insert('', tk.END, values=row)
            self.RtreeScrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.Rtree.yview)
            self.Rtree.configure(yscroll=self.RtreeScrollbar.set)
            self.RtreeScrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.Rtree.pack()

    def on_closing(self):
        if self.sensor_thread:
            self.sensor_thread.stop()
            self.sensor_thread.stop_event.set()
        #self.sensor_thread.join()
        self.root.destroy()

    def redrawFigs(self):
        self.figs.addSampleCanvas(self.charts_frame)
        #  Temporary Comment To Show A Sample Graph self.figs.axis.clear()
        self.figs.axis.clear()

        # self.datalist.append((self.curve1[:, 0], self.curve1[:, 1], ""))
        self.datalist.append((self.bzr_control_points[:, 0], self.bzr_control_points[:, 1], "ro:"))

        #  self.figs.updatePlot([1], self.datalist[0])
        # for datalist in self.datalistforgraph:
        #     self.figs.updatePlot(datalist[0], datalist[1])
        for datalist in self.datalist:
            self.figs.updatePlot(datalist[0], datalist[1], datalist[2])

    def UpdateFigs(self):
        self.figs.axis.clear()
        for datalist in self.datalistforgraph:
            self.figs.updatePlot([1], datalist)

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

    def removeFromList(self):
        pass

    def isFloat(self, digits:str):
        dgts = digits.replace(".", "")
        dgts = dgts.replace("-", "")
        if dgts.isdigit():
            return True
        else:
            return False

    def getNumbersListFromCommaSeparatedString(self, dta:str):
        lst = None
        if ',' in dta:
            lst = dta.split(',')
            for ele in lst:
                ele = ele.rstrip()
                # if ele.isdigit():
                if self.isFloat(ele):
                    pass
                else:
                    return [dta]

            return lst
        else:
            return [dta]

    def onSerialDataReceived(self, event):
        if event:
            #  data = event.__getattribute__("data")
            #  print(event.__dict__)
            #  self.serialdatalistbox.insert(self.serialdatalistbox.size(), "%s" % event)
            temp_list = []
            dta = "%s" % event
            try:
                lst = self.getNumbersListFromCommaSeparatedString(dta)
                lstsize = len(lst)
                #  print(f"csv size :{dta} : {str(lstsize)}")

                temp_list.clear()
                ####  if float(dta) < 10:
                for idx in range(lstsize):
                    temp_list.append(float(str(lst[idx])))
                    #  self.datalist[idx].append(float(str(lst[idx])))
                    #  print(str(idx))
                    #  print(str(self.datalist[-1]))
                self.datalist.append(temp_list)
                # if '\n' not in self.serdata:
                #     self.serdata += dta
                #     print(self.serdata)
                # else:
                #     self.datalist[0].append(float(str(self.serdata)))
                #     self.serdata = ''
                #  This Listbox is hidden and not used for now. I am concentrating to Treeview GUI for table
                #  view for multiple columns.
                #  self.serialdatalistbox.insert(self.serialdatalistbox.size(), float(str(dta)))
                self.reset_data_table_timer()
            except Exception as e:
                print(f"ERR::{str(e)}")
            txt = dta.replace('\r', '\n')
            self.textbox_rx.insert(tk.END, txt)
            ###  print(dta)
            #  self.serialdatalistbox.insert(self.serialdatalistbox.size(), "Event")
    def connect_to_port(self):
        selected_index = self.comportlisttree.curselection()
        #  print(selected_index) #  selected_index is a tuple like (0, )
        if selected_index:
            selected_port = self.com_ports[selected_index[0]] #  self.comportlisttree.get(selected_index)
            self.serial_port = selected_port
            try:
                ser = serial.Serial(selected_port, baudrate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0)

                #  timeout parameter in seconds for reading timeout. It is optional. If it is not given or
                #  None, read operation will block the execution.
                print(f"COM PORT Connected : {selected_port}")
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

                self.startCollect() # Just start the thread ( self.sensor_thread.start() )

            except serial.SerialException as e:
                print("Error Serial Port Connection", e)

    #  Write a command to the COM port (example: sending 'Hello')
    #  ser.write(b'Hello\n')  # The device connected to the COM port needs to understand this command


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dash = DASH()
    dash.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
