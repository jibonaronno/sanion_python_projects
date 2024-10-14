#  This code generated from chatGPT
import tkinter as tk
from tkinter import ttk
import threading
import time

class WorkerThread(threading.Thread):
    def __init__(self, event):
        threading.Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(1):
            # Simulate some work
            data = "Data from worker thread"
            # Update the GUI
            root.event_generate("<<DataAvailable>>", when="tail", data=data)

def update_gui(event):
    # Update the GUI with the data received from the worker thread
    label.config(text=event.data)

# Create the main window
root = tk.Tk()
root.title("Threaded Tkinter GUI")

# Create a label to display data from the thread
label = ttk.Label(root, text="Waiting for data...")
label.pack(padx=10, pady=10)

# Create an event to control the worker thread
stop_event = threading.Event()

# Create and start the worker thread
worker_thread = WorkerThread(stop_event)
worker_thread.start()

# Bind the DataAvailable event to the update_gui function
root.bind("<<DataAvailable>>", update_gui)

# Start the Tkinter event loop
root.mainloop()

# Stop the worker thread when the GUI is closed
stop_event.set()
