from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk

import serial
from serial.tools import list_ports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.btnConnect = tk.Button(self.frameA, text="Connect", command=self.connect_port, state="disabled")
        self.btnRead.pack(side="left")
        self.btnConnect.pack(side="left")
        self.charts_frame = tk.Frame(self.root, borderwidth=2, relief="raised")
        self.charts_frame.pack(side="bottom")
        self.upper_frame = tk.Frame(self.charts_frame)
        self.upper_frame.pack(fill="both", expand=True)
        self.canvases = []

        self.comportlisttree = tk.Listbox(self.side_frame, width=50)
        self.comportlisttree.bind("<<ListboxSelect>>", self.on_select_list_item)
        self.comportlisttree.pack(padx=5, pady=5)
        self.list_com_ports()

    def show(self):
        self.root.mainloop()

    def list_com_ports(self):
        self.comportlisttree.delete(0, tk.END)
        com_ports = [port.device for port in list_ports.comports()]
        for port in com_ports:
            self.comportlisttree.insert(tk.END, port)

    def on_select_list_item(self, event):
        selected_index = self.comportlisttree.curselection()
        if selected_index:
            selected_port = self.comportlisttree.get(selected_index)
            print("Selected Port : ", selected_port)
            self.btnConnect.config(state="normal")

    def connect_to_port(self):
        selected_index = self.comportlisttree.curselection()
        if selected_index:
            selected_port = self.comportlisttree.get(selected_index)

            try:
                ser = serial.Serial(selected_port, 15200)
                print("COM PORT Connected")

            except serial.SerialException as e:
                print("Error Serial Port Connection", e)

    def connect_port(self):
        pass

    def readAgain(self):
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dash = DASH()
    dash.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
