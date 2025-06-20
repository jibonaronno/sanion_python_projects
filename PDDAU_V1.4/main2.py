#!/usr/bin/python3

import sys
import warnings
from os.path import join, dirname, abspath

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

from qtpy import uic
from qtpy.QtCore import Slot
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget
import qtmodern.styles
import qtmodern.windows
from collections import deque
import serial
import serial.tools.list_ports as port_list

from plotview import PlotView
from mimic import Mimic
from comparison_chart import CompareChartWidget
import os
import json

# Import the new JAX-optimized server (replaces old pddsrvr)
try:
    from pddsvr_jax import FastServerThread, benchmark_data_generation

    JAX_SERVER_AVAILABLE = True
    print("JAX-optimized server loaded successfully")
except ImportError as e:
    print(f"JAX server import error: {e}")
    print("Falling back to original server implementation")
    JAX_SERVER_AVAILABLE = False
    # Fallback to original server if available
    try:
        from pddsrvr import ServerThread as FastServerThread


        def benchmark_data_generation():
            print("Using original server - no benchmark available")
    except ImportError:
        print("No server implementation available!")
        FastServerThread = None
        benchmark_data_generation = None
import threading

os.environ["XDG_SESSION_TYPE"] = "xcb"
# _UI5 = join(dirname(abspath(__file__)), 'charttabs.ui')
_UI_TOP = join(dirname(abspath(__file__)), 'top.ui')


class Configs(object):
    def __init__(self):
        super().__init__()
        self.settings = None
        self.local_ip = ""
        self.local_port = ""
        self.remote_ip = ""
        self.remote_port = ""

    def loadJson(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.settings = json.load(f)
                self.local_ip = self.settings.get("local_ip", "")
                self.local_port = self.settings.get("local_port", "")
        except Exception as e:
            print(f"Error Loading JSON File main.py : {str(e)}")


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = uic.loadUi(_UI_TOP, self)
        self.mimic = Mimic(self.customa)
        # verticalLayout_4 = QVBoxLayout()
        # self.verticalLayout_4.addWidget(self.mimic)
        self.comparison_chart = None
        self.UiComponents()
        self.configs = Configs()
        self.configs.loadJson("settings.json")
        self.event_pddthread_stop = threading.Event()
        self.send_samples = threading.Event()

        self.plotview = PlotView()

        # Initialize the JAX-optimized server (with fallback)
        if FastServerThread is not None:
            self.server_thread = FastServerThread(host='192.168.10.147', port=5000)
            self.server_thread.received.connect(self.on_ServerThreadSignalCallback)
            self.server_available = True
        else:
            print("No server implementation available!")
            self.server_thread = None
            self.server_available = False

        # Add status tracking
        self.server_running = False
        self.data_generation_started = False

        try:
            self.qlist.addItem('Local IP : ' + self.configs.local_ip)
            self.qlist.addItem('Local Port : ' + self.configs.local_port)
            self.lineEdit.setText(self.configs.local_ip)
        except Exception as e:
            print(f'Error main.py : {str(e)}')

        # Run performance benchmark on startup
        self.run_startup_benchmark()

        self.show()

    def run_startup_benchmark(self):
        """Run JAX performance benchmark on startup to show improvements."""
        try:
            if JAX_SERVER_AVAILABLE and benchmark_data_generation is not None:
                print("=" * 60)
                print("JAX-OPTIMIZED SPECTRUM SERVER STARTING")
                print("=" * 60)
                benchmark_data_generation()
                print("=" * 60)
            else:
                print("=" * 60)
                print("SPECTRUM SERVER STARTING (STANDARD MODE)")
                print("=" * 60)
        except Exception as e:
            print(f"Benchmark failed: {e}")

    def UiComponents(self):
        self.actionOpen.triggered.connect(self.OpenFile)
        self.actionOpen_Folder.triggered.connect(self.OpenFolder)

    def OpenFile(self):
        print("Menu -> Open File")
        location = dirname(abspath(__file__)) + '\\'
        fname = QFileDialog.getOpenFileName(self, 'Open file', location, "json files (*.json *.txt)")

    def OpenFolder(self):
        print("Menu -> Open Folder")
        location = dirname(abspath(__file__)) + '\\'
        foldername = QFileDialog.getExistingDirectory(self, "Select Folder", location)
        self.comparison_chart = CompareChartWidget(foldername)
        self.comparison_chart.showNormal()

    def on_ServerThreadSignalCallback(self, message):
        """Enhanced callback handler for the JAX-optimized server."""
        print(f"[SERVER] {message}")

        if message == "showplot":
            self.plotview.showNormal()
            # Inject data stream to graph using the current batch data
            try:
                if hasattr(self.server_thread, 'current_batch') and self.server_thread.current_batch is not None:
                    # Use the first packet from the current batch for plotting
                    sample_data = self.server_thread.current_batch[0]
                    # Convert JAX array to numpy for plotting
                    import jax.numpy as jnp
                    sample_data_np = jnp.array(sample_data)
                    self.plotview.injectDataStreamToGraph_16bit(sample_data_np)
                else:
                    print("No batch data available for plotting yet")
            except Exception as e:
                print(f"Error injecting data to plot: {e}")

        elif "Generated new batch" in message:
            # Update plot with new batch data when available
            self.update_plot_with_current_batch()

        elif "New connection" in message:
            self.update_status_display(f"Client connected: {message}")

        elif "Connection closed" in message:
            self.update_status_display(f"Client disconnected: {message}")

    def update_plot_with_current_batch(self):
        """Update the plot view with current batch data."""
        try:
            if (hasattr(self.server_thread, 'current_batch') and
                    self.server_thread.current_batch is not None and
                    hasattr(self.plotview, 'injectDataStreamToGraph_16bit')):
                # Use the middle packet from the batch for a representative sample
                middle_index = len(self.server_thread.current_batch) // 2
                sample_data = self.server_thread.current_batch[middle_index]

                # Convert JAX array to numpy for plotting
                import jax.numpy as jnp
                sample_data_np = jnp.array(sample_data)
                self.plotview.injectDataStreamToGraph_16bit(sample_data_np)
        except Exception as e:
            print(f"Error updating plot: {e}")

    def update_status_display(self, message):
        """Update status display in the GUI."""
        try:
            # Add to the list widget if it exists
            if hasattr(self, 'qlist'):
                self.qlist.addItem(f"[{self.get_timestamp()}] {message}")
                # Keep only last 20 messages
                if self.qlist.count() > 20:
                    self.qlist.takeItem(0)
                # Scroll to bottom
                self.qlist.scrollToBottom()
        except Exception as e:
            print(f"Error updating status: {e}")

    def get_timestamp(self):
        """Get current timestamp for status messages."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    @Slot()
    def on_btnInit_clicked(self):
        """Initialize and start the JAX-optimized server."""
        if not self.server_available:
            self.update_status_display("Error: No server implementation available!")
            return

        if not self.server_running:
            server_type = "JAX-optimized" if JAX_SERVER_AVAILABLE else "standard"
            print(f"Starting {server_type} spectrum server...")
            self.update_status_display(f"Initializing {server_type} server...")

            try:
                self.server_thread.start()
                self.server_running = True
                self.update_status_display("Server started successfully")

                # Update button states if they exist
                if hasattr(self, 'btnInit'):
                    self.btnInit.setText("Server Running")
                    self.btnInit.setEnabled(False)
                if hasattr(self, 'btnStop'):
                    self.btnStop.setEnabled(True)

            except Exception as e:
                self.update_status_display(f"Error starting server: {e}")
                print(f"Error starting server: {e}")
        else:
            self.update_status_display("Server is already running")

    @Slot()
    def on_btnProcD_clicked(self):
        """Process data - legacy compatibility."""
        print("Process Data button clicked")
        self.update_status_display("Data processing enabled")

        # The JAX server automatically processes data, so this is mainly for UI feedback
        if self.server_running:
            self.data_generation_started = True
            self.update_status_display("High-speed JAX data generation active")
        else:
            self.update_status_display("Please start server first")

    @Slot()
    def on_btnStop_clicked(self):
        """Stop the server."""
        if not self.server_available:
            self.update_status_display("No server to stop")
            return

        if self.server_running:
            server_type = "JAX-optimized" if JAX_SERVER_AVAILABLE else "standard"
            print(f"Stopping {server_type} server...")
            self.update_status_display("Stopping server...")

            try:
                self.server_thread.stop()
                self.server_running = False
                self.data_generation_started = False
                self.update_status_display("Server stopped successfully")

                # Update button states if they exist
                if hasattr(self, 'btnInit'):
                    self.btnInit.setText("Initialize Server")
                    self.btnInit.setEnabled(True)
                if hasattr(self, 'btnStop'):
                    self.btnStop.setEnabled(False)

            except Exception as e:
                self.update_status_display(f"Error stopping server: {e}")
                print(f"Error stopping server: {e}")
        else:
            self.update_status_display("Server is not running")

    @Slot()
    def on_btnchk128_clicked(self):
        """Legacy 128 samples option - not applicable to JAX version."""
        if JAX_SERVER_AVAILABLE:
            self.update_status_display("JAX server uses optimized 1024 samples per packet")
            print("JAX server automatically uses optimal sample size (1024)")
        else:
            self.update_status_display("Using standard server configuration")
            print("Standard server configuration")

    def closeEvent(self, event):
        """Stop the server thread when closing the window."""
        print("Application closing...")
        if self.server_available and self.server_running:
            print("Stopping server thread...")
            self.server_thread.stop()
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        from PyQt5.QtCore import Qt

        if event.key() == Qt.Key_F5:
            # F5 to restart server
            if self.server_running:
                self.on_btnStop_clicked()
            self.on_btnInit_clicked()

        elif event.key() == Qt.Key_F12:
            # F12 to run benchmark
            self.run_startup_benchmark()

        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # qtmodern.styles.dark(app)
    qtmodern.styles.light(app)

    # Create main window
    mw_class_instance = MainWindow()
    mw = qtmodern.windows.ModernWindow(mw_class_instance)

    # Add window title to reflect optimization status
    if JAX_SERVER_AVAILABLE:
        mw.setWindowTitle("Spectrum Server - JAX Optimized")
    else:
        mw.setWindowTitle("Spectrum Server - Standard Mode")

    # Show window
    mw.showNormal()

    sys.exit(app.exec_())