# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from pyModbusTCP.server import ModbusServer, DataBank
from pyModbusTCP.client import ModbusClient
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import threading
from threading import Timer
from os.path import join, dirname, abspath

class RepeatedTimer(object):
    def __init__(self, interval, function):
        self._timer = None
        self.interval = interval
        self.function = function
        #self.args = args
        #self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function()

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

class DASH(object):
    def __init__(self, _parent):
        self.parent = _parent
        self.root = tk.Tk()
        self.root.title("DASH")
        self.root.state("zoomed")
        self.side_frame = tk.Frame(self.root)  #,yscrollcommand=scrollbar.set)
        self.side_frame.pack(side="left", fill="y")
        self.label = tk.Label(self.side_frame, text="Dashboard", bg="#4C2A85", fg="#FFF", font=25)
        self.label.pack(pady=50, padx=20)
        self.btnRead = tk.Button(self.side_frame, text="Read Again", command=self.readAgain)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.btnRead.pack()
        self.btnReadEventHeader = tk.Button(self.side_frame, text='Read Event Header', command=self.readEventHeader)
        self.btnReadEventHeader.pack()
        self.charts_frame = tk.Frame(self.root)
        self.charts_frame.pack()
        self.upper_frame = tk.Frame(self.charts_frame)
        self.upper_frame.pack(fill="both", expand=True)
        self.canvases = []
        self._timer = None
        self.rt = RepeatedTimer(10, self.parent.readAllAndSave)


    def readAgain(self):
        self.parent.readAgain()

    def readEventHeader(self):
        self.parent.readAllAndSave()

    def assignCanvases(self, plots: []):
        plot_counts = len(plots)
        print("PLOTS COUNTS : ", plot_counts)
        if plot_counts > 0:
            for plot in plots:
                print(" len(plot) : ", len(plot))
                self.canvases.append(FigureCanvasTkAgg(plot[0], self.upper_frame))

    def drawCanvases(self):
        for canvas in self.canvases:
            canvas.draw()
            canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def regularCollectAcq(self):
        # self._timer = threading.Timer(6.0, self.parent.readAllAndSave).start()
        self._timer = threading.Timer(8.0, self.regularCollectAcq).start()
        self.parent.readAllAndSave()
    def show(self):
        # self.regularCollectAcq()
        #threading.Timer(15.0, self.regularCollectAcq).start()
        self.root.mainloop()

    def on_closing(self):
        try:
            # self._timer.cancel()
            self.rt.stop()
            self.root.destroy()
        except Exception as e:
            print(f'{str(e)}')
        print("Root Destroyed")
        exit(0)

class SRVR(object):
    def __init__(self):
        _fpath = join(dirname(abspath(__file__)), 'config.txt')
        _ipaddress = self.readIpaddressFromFile(_fpath)
        self._savepath = self.readPathFromFile(_fpath)
        print(f'ACQ IP ADDRESS: {_ipaddress}')
        #self.srvr = ModbusServer("localhost", 100, no_block=False)
        # self.clnt = ModbusClient(host=_ipaddress[:-1], port=100, auto_open=False, debug=False)
        self.clnt = ModbusClient(host=_ipaddress[:-1], port=100, auto_open=False)
        #self.__address = 8212
        self.__address = 10516
        #self.__address = 17428
        self.start_address = self.__address
        self.voltwave = []
        self.subplots = []
        self.subplot_lines = []
        self.dash = DASH(self)
        self.single_reg =  0x0000
        self.contactAB_string = ""
        self.isPlotAssigned = False
        for _i in range(7):
            #xrr = np.array([1, 2, 3, 4, 5, 6])
            #yrr = np.array([1, 2, 3, 4, 5, 6])
            xrr = np.linspace(0, 1800, 8)
            yrr = np.linspace(-4000, 18000, 8)
            self.subplots.append(plt.subplots())
            self.subplot_lines.append(self.subplots[_i][1].plot(xrr, yrr)[0])
        for plot in self.subplots:
            xrr = np.array([0, 0])
            yrr = np.array([0, 0])
            #plot[1].plot(xrr, yrr)
            plot[1].set_xlabel('X Axis Data')
            plot[1].set_ylabel('Y Axis')
            plot[0].canvas.draw()
            plot[0].canvas.flush_events()

    def readIpaddressFromFile(self, _filename):
        with open(_filename, 'r') as f:
            lines = f.readlines()
            return lines[0]

    def readPathFromFile(self, _filename):
        with open(_filename, 'r') as f:
            lines = f.readlines()
            return lines[1]

    def twos_comp(self, val, bits):
        """compute the 2's complement of int value val"""
        if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)        # compute negative value
        return val                         # return positive value as is

    def plotme(self, xarr, yarr, title):
        xr = np.array(xarr)
        yr = np.array(yarr)
        plot = (None, None)
        plot = plt.subplots() #plt.plot(xr,yr, 'r')
        plot[1].plot(xr, yr)
        plot[1].set_xlabel('X Axis Data')
        plot[1].set_ylabel('Y Axis')
        plot[1].set_title(title)
        plot[0].canvas.draw()
        plot[0].canvas.flush_events()
        return plot

    def plotme2(self, plot, line, xar, yar, title):
        xr = np.array(xar)
        yr = np.array(yar)
        #line, = plot[1].plot(xr, yr)
        plot[1].set_xlabel('X Axis Data')
        plot[1].set_ylabel('Y Axis')
        plot[1].set_title(title)
        line.set_xdata(xr)
        line.set_ydata(yr)
        plot[0].canvas.draw()
        plot[0].canvas.flush_events()


    def assignPlotsToDash(self):
        self.dash.canvases.clear()
        self.dash.assignCanvases(self.subplots)
        self.dash.drawCanvases()
        self.isPlotAssigned = True

    def cnnct(self):
        try:
            #self.srvr.start()
            ix = 0
            self.clnt.open()
            #while True:
            while True:
                regs_1 = self.clnt.read_holding_registers(self.__address, 64)
                for reg in regs_1:
                    self.voltwave.append(str(self.__address + ix) + "," + str(reg) + "\n")
                self.__address = self.__address + 64
                if self.__address > (self.start_address + (35 * 64)):
                    break
            pprint(regs_1)
            with open("voltwave.csv", "w") as csvfile:
                for line in self.voltwave:
                    csvfile.write("".join(line))
            self.clnt.close()
        except Exception as e:
            #self.srvr.stop()
            self.clnt.close()
            print("Exception " + str(e))

    def getFilenameTimeStringFromEpoch(self, epoch:int):
        strfil = '22-' + datetime.datetime.fromtimestamp(epoch).strftime('%Y%m%d%H%M%S') + '.dat'
        return strfil

    # Wave data coming from ACQ in 16bit format where the Value is +32767 to -32767 . But since our variable is 32 bit
    # wide, it cannot detect Negative values. So by checking the flag is_twoscompl, it also check if the value is greater
    # than 32767, it will convert it to negative value. Similar code is applied to the LU code and Python acq code.
    # Here 3rd Parameter Should be BlockSize instead _reg_count
    def readRegisters(self, _start_address, _reg_count, _block_count, _filename='N/A', is_twoscompl=False, is_plot=False, plot_index=0):
        try:
            xaxis_arr = []
            yaxis_arr = []
            start_address = _start_address
            total_size = 0
            ix = 0
            dx = ix
            self.clnt.open()
            self.voltwave.clear()
            self.voltwave.append("Reg Address,Data\n")
            while True:
                regs_1 = self.clnt.read_holding_registers(_start_address, _reg_count)
                # print(type(regs_1))
                if len(regs_1) > 0:
                    self.single_reg = regs_1[0]
                total_size = total_size + len(regs_1)
                for reg in regs_1:
                    if(is_twoscompl):
                        if reg > 0x7FFF:
                            self.voltwave.append(str(_start_address + ix) + "," + str(((reg & 0x7FFF) - 0x7FFF)) + "\n")
                        else:
                            self.voltwave.append(str(_start_address + ix) + "," + str(reg) + "\n")
                            #self.voltwave.append(str(_start_address + ix) + "," + str(self.twos_comp(reg, 16)) + "\n")
                    else:
                        self.voltwave.append(str(_start_address + ix) + "," + str(reg) + "\n")
                    xaxis_arr.append(dx)
                    if reg > 0xEFFF:
                        yaxis_arr.append((reg & 0x7FFF) - 0x7FFF)
                    else:
                        yaxis_arr.append(reg)
                    ix = ix + 1
                    dx = dx + 1
                _start_address = _start_address + _reg_count
                ix = 0
                if _block_count == 0:
                    break
                if _start_address > (start_address + ((_block_count - 1) * _reg_count)):
                    break
            #pprint(regs_1)
            print(" SIZE : ", str(total_size))
            if _filename != "NA":
                with open(_filename, "w") as csvfile:
                    for line in self.voltwave:
                        csvfile.write("".join(line))
            self.clnt.close()
            self.voltwave.clear()
            # if is_plot:
            #     self.plotme2(self.subplots[plot_index], self.subplot_lines[plot_index], xaxis_arr, yaxis_arr, title=_filename)
            #     return 1
            # else:
            #     return self.single_reg
            return self.single_reg
        except Exception as e:
            #self.srvr.stop()
            self.clnt.close()
            self.voltwave.clear()
            print("Exception " + str(e))
            return None

    def readRegistersBytes(self, _start_address, _block_size, _number_of_blocks):
        try:
            start_address = _start_address
            total_size = 0
            ix = 0
            dx = ix
            _bytes = bytearray(b'')
            self.clnt.open()
            while True:
                regs_1 = self.clnt.read_holding_registers(_start_address, _block_size)
                for reg in regs_1:
                    bytes_read = reg.to_bytes(2, "little")
                    _bytes.extend(bytes_read)
                _start_address = _start_address + _block_size
                dx = dx + 1
                if dx == _number_of_blocks:
                    break
            self.clnt.close()
            return _bytes
        except Exception as e:
            self.clnt.close()
            print("Exception " + str(e))
            return None

    def readRecordCount(self):
        if self.clnt.open():
            reg = self.readRegisters(1028, 1, 0, "NA")
            print(" STATUS       = ", str(reg))
            print(" STATUS       = ", str(hex(reg)))
            print(f" RECORD COUNT = {str((0x00FF & reg))}")
            self.clnt.close()
            self.voltwave.clear()
            return (0x00FF & reg)
        return 0

    def readEvent(self):
        _bytes_arr = self.readRegistersBytes(1200, 34, 1)
        print(f'_bytes_arr size : {len(_bytes_arr)}')
        lidx = 0
        for _byte in _bytes_arr:
            if lidx < 8:
                print(f'{hex(_byte)}', end=' ')
                lidx = lidx + 1
            else:
                print(f'\n{hex(_byte)}', end=' ')
                lidx = 1

    def readAllAndSave(self):
        if(self.readRecordCount() > 0):
            fullBytesArr = self.readFullBytes()
            tsize = 0
            for _bytes in fullBytesArr:
                tsize = tsize + len(_bytes)
            epoc_time = int.from_bytes(fullBytesArr[0][4:7], "little")
            print(f'Total Size : {tsize}')
            print(f'EPOC : {epoc_time}')
            print(f'Filename: {self.getFilenameTimeStringFromEpoch(epoc_time)}')
            #_filename = 'C:\\test1\\' + self.getFilenameTimeStringFromEpoch(epoc_time)
            _filename = self._savepath[:-1] + self.getFilenameTimeStringFromEpoch(epoc_time)
            print(f'SAVE FILE FULL PATH: {_filename}')
            with open(_filename, "wb") as binfile:
                for _bytes in fullBytesArr:
                    binfile.write(_bytes)

    def readFullBytes(self):
        _bytes_arr = []
        self.readRecordCount()
        _bytes_arr.append(self.readRegistersBytes(1200, 34, 1))
        _bytes_arr.append(self.readRegistersBytes(1300, 64, 36))
        _bytes_arr.append(self.readRegistersBytes(3604, 64, 36))
        _bytes_arr.append(self.readRegistersBytes(5908, 64, 36))
        self.readRegistersBytes(8212, 64, 36)
        _bytes_arr.append(self.readRegistersBytes(10516, 64, 36))
        _bytes_arr.append(self.readRegistersBytes(12820, 64, 36))
        _bytes_arr.append(self.readRegistersBytes(15124, 64, 36))
        _bytes_arr.append(self.readRegistersBytes(17428, 64, 18))
        return _bytes_arr

    def readInitiateAndContact(self):
        subplt = self.readRegisters(17428, 64, 18, "init_and_contact.csv", is_plot=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readTripCoil1(self):
        self.readRegisters(1300, 64, 36, "trip1.csv", is_plot=True, plot_index=0, is_twoscompl=True)

    def readTripCoil2(self):
        self.readRegisters(3604, 64, 36, "trip2.csv", is_plot=True, plot_index=1, is_twoscompl=True)

    def readPhaseACurr(self):
        subplt = self.readRegisters(10516, 64, 36, "PhaseA_Curr.csv", is_plot=True, plot_index=2, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readPhaseBCurr(self):
        subplt = self.readRegisters(12820, 64, 36, "PhaseB_Curr.csv", is_plot=True, plot_index=3, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readPhaseCCurr(self):
        subplt = self.readRegisters(15124, 64, 36, "PhaseC_Curr.csv", is_plot=True, plot_index=4, is_twoscompl=True)
        #subplt = self.readRegisters(15124, 64, 36, "PhaseC_Curr.csv") #, is_plot=True, plot_index=4, is_twoscompl=True)

    def readVoltWave(self):
        subplt = self.readRegisters(8212, 64, 36, "voltwave.csv", is_plot=True, plot_index=5, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readInitContact(self):
        pass
        #subplt = self.readRegisters(17428, 64, 18, "InitContact.csv", is_plot=False, plot_index=5, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readCloseCoil(self):
        subplt = self.readRegisters(5908, 64, 36, "closecoil.csv", is_plot=True, plot_index=4, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readAgain(self):
        #self.dash.canvases.clear()
        #self.subplots.clear()
        for plot in self.subplots:
            plot[0].canvas.flush_events()
        if (self.readRecordCount() > 0):
            self.readTripCoil1()
            self.readTripCoil2()
            self.readCloseCoil()
            self.readVoltWave()
            #self.readInitContact()
            self.readPhaseACurr()
            self.readPhaseBCurr()
            self.readPhaseCCurr()
            self.readInitiateAndContact()
            if not self.isPlotAssigned:
                self.assignPlotsToDash()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    srvr = SRVR()

    if(srvr.readRecordCount() > 0):
        srvr.readTripCoil1()
        srvr.readTripCoil2()
        srvr.readCloseCoil()
        srvr.readVoltWave()
        srvr.readInitContact()
        srvr.readPhaseACurr()
        srvr.readPhaseBCurr()
        srvr.readPhaseCCurr()
        srvr.readInitiateAndContact()
        srvr.assignPlotsToDash()
    srvr.dash.show()
