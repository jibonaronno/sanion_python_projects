

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class POLARBLOCK(object):
    def __init__(self):
        self.fig = Figure(figsize=(10, 2), dpi=120)
        self.axis1 = self.fig.add_subplot(1, 1, 1, projection='polar')
        self.canvases = []

    def addCanvas(self, fig, frame):
        canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvases.append(canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def updatePolarPlot(self, length, angle):
        x = None
        xarr = np.linspace(0, length, length)
        x = np.array(xarr)
        theta = np.linspace(angle, angle, length)

        self.axis1.plot(theta, x)

        for canvas in self.canvases:
            canvas.draw()

    def addSampleCanvas(self, frame):
        new_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # new_y = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
        new_y = np.linspace(0.3, 0.3, len(new_x))
        self.axis1.clear()
        self.axis1.plot(new_y, new_x)
        new_y = np.linspace((120 * 0.017453), (120 * 0.017453), len(new_x))
        self.axis1.plot(new_y, new_x)
        new_y = np.linspace((-120 * 0.017453), (-120 * 0.017453), len(new_x))
        self.axis1.plot(new_y, new_x)
        self.addCanvas(self.fig, frame)

    def updatePlot(self, new_x:[], new_y:[], angle:float):
        x = []
        if len(new_y) > 0:
            if len(new_x) != len(new_y):
                xarr = np.linspace(1, len(new_y), len(new_y))
                x = np.array(xarr)
            else:
                x = np.array(new_x)
            y = np.array(new_y)
            print(f"LEN x:{len(x)} LEN y:{len(y)}")
            #  self.plot.clear()
            self.axis1.plot(x, angle)

            for canvas in self.canvases:
                canvas.draw()
