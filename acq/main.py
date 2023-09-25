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


class DASH(object):
    def __init__(self, _parent):
        self.parent = _parent
        self.root = tk.Tk()
        self.root.title("DASH")
        self.root.state("zoomed")
        self.side_frame = tk.Frame(self.root) #, yscrollcommand=scrollbar.set)
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

    def readAgain(self):
        self.parent.readAgain()


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
    def show(self):
        self.root.mainloop()

class SRVR(object):
    def __init__(self):
        #self.srvr = ModbusServer("localhost", 100, no_block=False)
        self.clnt = ModbusClient(host='192.168.247.100', port=100, auto_open=False, debug=False)
        #self.__address = 8212
        self.__address = 10516
        #self.__address = 17428
        self.start_address = self.__address
        self.voltwave = []
        self.subplots = []
        self.subplot_lines = []
        self.dash = DASH(self)
        self.single_reg =  0x0000
        for _i in range(7):
            #xrr = np.array([1, 2, 3, 4, 5, 6])
            #yrr = np.array([1, 2, 3, 4, 5, 6])
            xrr = np.linspace(0, 1800, 8)
            yrr = np.linspace(-4000, 5000, 8)
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

    def readRegister(self, _start_address, is_twoscompl=False, is_plot=False):
        pass
    def readRegisters(self, _start_address, _reg_count, _block_count, _filename, is_twoscompl=False, is_plot=False, plot_index=0):
        try:
            #self.single_reg = 0x0000
            xaxis_arr = []
            yaxis_arr = []
            start_address = _start_address
            total_size = 0
            ix = 1
            ix = 0
            dx = ix
            self.clnt.open()
            self.voltwave.clear()
            self.voltwave.append("Reg Address,Data\n")
            while True:
                regs_1 = self.clnt.read_holding_registers(_start_address, _reg_count)
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
            if is_plot:
                self.plotme2(self.subplots[plot_index], self.subplot_lines[plot_index], xaxis_arr, yaxis_arr, title=_filename)
                return 1
            else:
                return self.single_reg
        except Exception as e:
            #self.srvr.stop()
            self.clnt.close()
            self.voltwave.clear()
            print("Exception " + str(e))
            return None

    def readRecordCount(self):
        self.clnt.open()
        reg = self.readRegisters(1028, 1, 0, "NA")
        print(" STATUS = ", hex(reg))
        self.clnt.close()
        self.voltwave.clear()
        return (0x00FF & reg)
    def readInitiateAndContact(self):
        subplt = self.readRegisters(17428, 64, 18, "init_and_contact.csv", is_plot=False)
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
        #subplt = self.readRegisters(15124, 64, 36, "PhaseC_Curr.csv", is_plot=True, plot_index=4, is_twoscompl=True)
        subplt = self.readRegisters(15124, 64, 36, "PhaseC_Curr.csv") #, is_plot=True, plot_index=4, is_twoscompl=True)

    def readVoltWave(self):
        subplt = self.readRegisters(8212, 64, 36, "voltwave.csv", is_plot=False, plot_index=5, is_twoscompl=True)
        #if subplt is not None:
            #self.subplots.append(subplt)

    def readInitContact(self):
        subplt = self.readRegisters(17428, 64, 18, "InitContact.csv", is_plot=True, plot_index=5, is_twoscompl=True)
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
            self.readInitContact()
            self.readPhaseACurr()
            self.readPhaseBCurr()
            self.readPhaseCCurr()
            self.readInitiateAndContact()


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
