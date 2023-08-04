# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from pathlib import Path
from os.path import join, dirname, abspath
import struct
from struct import unpack
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle

class SubPlots(object):
    def __init__(self):
        self.plots = []
        self.indx = 1

    def addPlot(self, yarr):
        #xarr = np.arange(0, len(yarr), 1)
        xarr = np.linspace(1, len(yarr), len(yarr))
        plotA = plt.subplot(2, 3, self.indx)
        self.indx = self.indx + 1
        print("LEN X : ", str(len(xarr)), " LEN Y : ", str(len(yarr)))
        x = np.array(xarr)
        y = np.array(yarr)
        print("LEN X : ", str(x.shape), " LEN Y : ", str(y.shape))
        plt.plot(x,y)
        self.plots.append(plotA)

    def Show(self):
        plt.show()
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

class BinaryFileReader(object):
    def __init__(self):
        self.ficontent = None
        self.str_elements = ["Event type                         : ",
                            "Event Time - year month day ...    : ",
                            "Event Time ms                      : ",
                            "Alert Level                        : ",
                            "Contact Duty A. Breaking Current A : ",
                            "Contact Duty B. Breaking Current B : ",
                            "Contact Duty C. Breaking Current C : ",
                            "Commulative Current A              : ",
                            "Commulative Current B              : ",
                            "Commulative Current C              : ",
                            "t1 integral                        : ",
                            "t1 max                             : ",
                            "t1 fem time                        : ",
                            "t2 integral                        : ",
                            "t2 max                             : ",
                            "t2 fem time                        : ",
                            "close coil integral                : ",
                            "close coil max                     : ",
                            "close fem time                     : ",
                            "Acontact op time                   : ",
                            "Bcontact op time                   : ",
                            "Block By phase A / input time      : ",
                            "Block By phase B / input time      : ",
                            "Block By phase C / input time      : ",
                            "Op Count                           : ",
                            "Sample Per Cycle                   : ",
                            "Cycle Count                        : "]
        pass

    def readFile(self, finame):
        with open(finame, mode='rb') as file:  # b is important -> binary
            self.ficontent = file.read()

    def readFil(self, finame):
        filname = join(dirname(abspath(__file__)), finame)
        print("File name: ", filname)
        self.ficontent = Path(filname).read_bytes()  # Python 3.5+

    def printContent_16(self, _offset=98):
        barray = self.ficontent[_offset:]
        print("Length : ", len(barray))
        #integers = struct.unpack("!%sH" % (len(barray) // 2), barray)
        integers = unpack('<'+'h'*(len(barray)//2), barray)
        print("Type of each element : ", type(integers[0]))
        print(integers)
        pass

    def getHeader(self):
        header = {}
        hdr_array = [] #bytearray(0)

        hdr_array.append(unpack(">B", self.ficontent[0:1])[0]) # Event type
        hdr_array.append(unpack("<I", self.ficontent[1:5])) # Event Time - year month day ...
        hdr_array.append(unpack(">I", self.ficontent[5:9])[0]) # Event Time ms
        hdr_array.append(unpack(">B", self.ficontent[9:10])[0])  # Alert Level
        hdr_array.append(unpack(">f", self.ficontent[10:14])[0])  # Contact Duty A. Breaking Current A
        hdr_array.append(unpack(">f", self.ficontent[14:18])[0])  # Contact Duty B. Breaking Current B
        hdr_array.append(unpack(">f", self.ficontent[18:22])[0])  # Contact Duty C. Breaking Current C
        hdr_array.append(unpack(">f", self.ficontent[22:26])[0])  # Commulative Current A
        hdr_array.append(unpack(">f", self.ficontent[26:30])[0])  # Commulative Current B
        hdr_array.append(unpack(">f", self.ficontent[30:34])[0])  # Commulative Current C
        hdr_array.append(unpack(">f", self.ficontent[34:38])[0])  # t1 integral
        hdr_array.append(unpack(">f", self.ficontent[38:42])[0])  # t1 max
        hdr_array.append(unpack(">f", self.ficontent[42:46])[0])  # t1 fem time
        hdr_array.append(unpack(">f", self.ficontent[46:50])[0])  # t2 integral
        hdr_array.append(unpack(">f", self.ficontent[50:54])[0])  # t2 max
        hdr_array.append(unpack(">f", self.ficontent[54:58])[0])  # t2 fem time
        hdr_array.append(unpack(">f", self.ficontent[58:62])[0])  # close coil integral
        hdr_array.append(unpack(">f", self.ficontent[62:66])[0])  # close coil max
        hdr_array.append(unpack(">f", self.ficontent[66:70])[0])  # close fem time
        hdr_array.append(unpack(">f", self.ficontent[70:74])[0])  # Acontact op time
        hdr_array.append(unpack(">f", self.ficontent[74:78])[0])  # Bcontact op time
        hdr_array.append(unpack(">f", self.ficontent[78:82])[0])  # Block By phase A / input time
        hdr_array.append(unpack(">f", self.ficontent[82:86])[0])  # Block By phase B / input time
        hdr_array.append(unpack(">f", self.ficontent[86:90])[0])  # Block By phase C / input time
        hdr_array.append(unpack(">I", self.ficontent[90:94])[0])  # Op Count
        hdr_array.append(unpack(">H", self.ficontent[94:96])[0])  # Sample Per Cycle
        hdr_array.append(unpack(">H", self.ficontent[96:98])[0])  # Cycle Count

        '''
        self.ficontent[0:1])[0]) # Event type
        self.ficontent[1:5])[0]) # Event Time - year month day ...
        self.ficontent[5:9])[0]) # Event Time ms
        self.ficontent[9:10])[0])  # Alert Level
        self.ficontent[10:14])[0])  # Contact Duty A. Breaking Current A
        self.ficontent[14:18])[0])  # Contact Duty B. Breaking Current B
        self.ficontent[18:22])[0])  # Contact Duty C. Breaking Current C
        self.ficontent[22:26])[0])  # Commulative Current A
        self.ficontent[26:30])[0])  # Commulative Current B
        self.ficontent[30:34])[0])  # Commulative Current C
        self.ficontent[34:38])[0])  # t1 integral
        self.ficontent[38:42])[0])  # t1 max
        self.ficontent[42:46])[0])  # t1 fem time
        self.ficontent[46:50])[0])  # t2 integral
        self.ficontent[50:54])[0])  # t2 max
        self.ficontent[54:58])[0])  # t2 fem time
        self.ficontent[58:62])[0])  # close coil integral
        self.ficontent[62:66])[0])  # close coil max
        self.ficontent[66:70])[0])  # close fem time
        self.ficontent[70:74])[0])  # Acontact op time
        self.ficontent[74:78])[0])  # Bcontact op time
        self.ficontent[78:82])[0])  # Block By phase A / input time
        self.ficontent[82:86])[0])  # Block By phase B / input time
        self.ficontent[86:90])[0])  # Block By phase C / input time
        self.ficontent[90:94])[0])  # Op Count
        self.ficontent[94:96])[0])  # Sample Per Cycle
        self.ficontent[96:98])[0])  # Cycle Count
        '''

        '''
        hdr_array.extend(self.ficontent[0:1])  # Event type
        hdr_array.extend(self.ficontent[5:1])  # Event Time - year month day ...
        hdr_array.extend(self.ficontent[9:5])  # Event Time ms
        hdr_array.extend(self.ficontent[9:10])  # Alert Level
        hdr_array.extend(self.ficontent[14:10])  # Contact Duty A. Breaking Current A
        hdr_array.extend(self.ficontent[18:14])  # Contact Duty B. Breaking Current B
        hdr_array.extend(self.ficontent[22:18])  # Contact Duty C. Breaking Current C
        hdr_array.extend(self.ficontent[26:22])  # Commulative Current A
        hdr_array.extend(self.ficontent[30:26])  # Commulative Current B
        hdr_array.extend(self.ficontent[34:30])  # Commulative Current C
        hdr_array.extend(self.ficontent[38:34])  # t1 integral
        hdr_array.extend(self.ficontent[42:38])  # t1 max
        hdr_array.extend(self.ficontent[46:42])  # t1 fem time
        hdr_array.extend(self.ficontent[50:46])  # t2 integral
        hdr_array.extend(self.ficontent[54:50])  # t2 max
        hdr_array.extend(self.ficontent[58:54])  # t2 fem time
        hdr_array.extend(self.ficontent[62:58])  # close coil integral
        hdr_array.extend(self.ficontent[66:62])  # close coil max
        hdr_array.extend(self.ficontent[70:66])  # close fem time
        hdr_array.extend(self.ficontent[74:70])  # Acontact op time
        hdr_array.extend(self.ficontent[78:74])  # Bcontact op time
        hdr_array.extend(self.ficontent[82:78])  # Block By phase A / input time
        hdr_array.extend(self.ficontent[86:82])  # Block By phase B / input time
        hdr_array.extend(self.ficontent[90:86])  # Block By phase C / input time
        hdr_array.extend(self.ficontent[94:90])  # Op Count
        hdr_array.extend(self.ficontent[96:94])  # Sample Per Cycle
        hdr_array.extend(self.ficontent[98:96])  # Cycle Count
        '''
        return hdr_array
        pass

    def convertContent(self):
        carr = bytearray(0)
        barr = self.ficontent[98:]
        length = int(len(barr)/2)
        for i in range(length):
            carr.extend([barr[i+1], barr[i]])
        return carr

    def saveToLittleEndian(self, filename):
        header = self.getHeader()
        print("Type : ", type(header), " Len : ", len(header))
        carr = self.convertContent()
        print(" Type : ", type(carr), " Size : ", len(carr))
        header.extend(carr)
        with open(filename, "wb") as binary_file:
            # Write bytes to file
            binary_file.write(header)
        pass

    def saveToCsv(self, content, filename):
        with open(filename, "w") as text_file:
            text_file.write(content)

class Viewer(object):
    def __init__(self):
        self.dash = DASH(self)
        self.frdr = BinaryFileReader()

    def Show(self):
        self.dash.show()

def ArraySwap16(arr):
    nrr = []
    for ele in arr:
        nrr.append((ele >> 8) | (ele << 8))
    return nrr

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    frdr = BinaryFileReader()
    frdr.readFil('01_22_20230804153352.dat') #("01_22_20230706160141.dat")
    #frdr.readFil("01_22_20230706160141.dat")

    frdr.getHeader()

    print(" Cycle Count : ", unpack("<H", frdr.ficontent[96:98])[0])
    print(" Cycle Count Hex 0x{0:04x}".format(unpack("<H", frdr.ficontent[96:98])[0]))

    hdrr = frdr.getHeader()
    lidxA = 0
    for ele in frdr.str_elements:
        print(ele, " ", hdrr[lidxA])
        lidxA += 1

    csv = ''

    #frdr.saveToLittleEndian("K_S623_GLU101_CH05_CBOP_3551284_22_20230712031752.converted.dat")

    #frdr.readFil("01_22_19991129150003.dat")
    #frdr.printContent_16()
    '''
    File index should start from 98. But index from 98 is showing some wrong value at the 
    beginning of the graph. 
    '''
    idxA = 104
    idxB = idxA + 4608
    plotter = SubPlots()
    barray = frdr.ficontent[idxA:idxA+4608]
    integers = unpack('>' + 'h' * (len(barray) // 2), barray) #Source is big Endian.
    plotter.addPlot(integers)

    # csv += "Trip1\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n,'

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA+4608]
    integers2 = unpack('>' + 'h' * (len(barray) // 2), barray)
    plotter.addPlot(integers2)

    # csv += "Trip2\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n,'

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA + 4608]
    integers3 = unpack('>' + 'h' * (len(barray) // 2), barray)
    plotter.addPlot(integers3)

    # csv += "Close\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n,'

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA + 4608]
    integers4 = unpack('>' + 'h' * (len(barray) // 2), barray)
    plotter.addPlot(integers4)

    # csv += "PhaseA\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n,'

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA + 4608]
    integers5 = unpack('>' + 'h' * (len(barray) // 2), barray)
    plotter.addPlot(integers5)

    # csv += "PhaseB\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n,'

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA + 4608]
    integers6 = unpack('>' + 'h' * (len(barray) // 2), barray)
    plotter.addPlot(integers6)

    idxA += 4608
    barray = frdr.ficontent[idxA:idxA + 2304]
    print(len(barray))
    integers7 = unpack('>' + 'b' * (len(barray)), barray)
    #plotter.addPlot(integers6)

    # csv += "PhaseC\n"
    # for i in integers:
    #     csv += str(i) + '\n'
    # csv += '\n'

    csv = "Trip1,Trip2,Close,PhaseA,PhaseB,PhaseC,Contact\n"

    # for i in range(len(integers)):
    #     csv += str(integers[i]) + "," + str(integers2[i]) + "," + str(integers3[i]) + "," + str(integers4[i]) + "," + str(integers5[i]) + "," + str(integers6[i]) + "," + str(integers7[i]) + "\n"
    #
    # frdr.saveToCsv(csv, "01_22_20230731171131.csv")

    plotter.Show()

    #view = Viewer()
    #view.Show()

    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
