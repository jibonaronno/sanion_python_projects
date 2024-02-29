from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from serial.tools import list_ports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DASH(object):
    def __init__(self):
        #  self.parent = _parent
        self.root = tk.Tk()
        self.root.title("DASH")
        self.root.state("zoomed")
        self.side_frame = tk.Frame(self.root)  #,yscrollcommand=scrollbar.set)
        self.side_frame.pack(side="left", fill="y")
        self.label = tk.Label(self.side_frame, text="Dashboard", bg="#4C2A85", fg="#FFF", font=25)
        self.label.pack(pady=50, padx=20)
        self.btnRead = tk.Button(self.side_frame, text="Read Again", command=self.readAgain)
        self.btnRead.pack()
        self.charts_frame = tk.Frame(self.root)
        self.charts_frame.pack()
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

    def readAgain(self):
        pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dash = DASH()
    dash.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
