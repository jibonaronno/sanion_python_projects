import sys
import serial
import pprint
import time
import enum
import queue
from queue import Queue
from os.path import join, dirname, abspath
from qtpy.QtCore import Slot, QTimer, QThread, Signal, QObject, Qt, QMutex

class PrimaryThread(QObject):
    signal = Signal(str)

    def __init__(self, serialPort, cmdlist):
        self.serialPort = serialPort
        # self.json = JsonSettings("settings.json")

        self.codelist = cmdlist
        self.flagStop = False
        self.pause = False
        super().__init__()

    def Stop(self):
        self.flagStop = True

    @Slot()
    def run(self):
        unit = []
        hexformat = ''
        inhex = ''
        while True:
            try:
                if self.pause:
                    time.sleep(100)
                    continue
                for line in self.codelist:
                    if self.flagStop:
                        break
                    # self.serialPort.reset_input_buffer()
                    #self.serialPort.write((str(line) + "\r\n").encode("utf-8"))
                    try:
                        print('Sending Cmd')
                        self.serialPort.write(line)
                        time.sleep(1)
                        in_waiting = self.serialPort.in_waiting
                        if in_waiting == 0:
                            time.sleep(1)
                            in_waiting = self.serialPort.in_waiting
                    except Exception as e:
                        print("serialPort.write(line) -- " + str(e))

                    jMessage = ""
                    try:
                        if in_waiting != 0:
                            unit = self.serialPort.read(in_waiting)
                    except Exception as e:
                        print('Ex in sensor Thread readline() 49 : ' + str(e))
                    for hx in unit:
                        hexformat = hexformat + '{0:02X} '.format(hx)
                    for hx in line:
                        inhex = inhex + '{0:02X} '.format(hx)
                    unit = b''
                    #self.signal.emit(str(line).format("")+ " - " + hexformat)
                    self.signal.emit(inhex + "- " + hexformat)
                    hexformat = ''
                    inhex = ''

            except serial.SerialException as ex:
                print("Error In SerialException" + ex.strerror)
                self.signal.emit("Stopped")
            except Exception as e:
                pprint.pprint(e)
                self.signal.emit("Stopped")

class SensorThread(QObject):
    signal = Signal(str)
    plst = []

    def __init__(self, serialPort, que):
        self.pressureque = que
        self.serialport = serialPort
        self.flagStop = False
        self.jMessage = ""
        self._beep = False
        self.flag_sensorlimit_tx = False
        self.strdata = ""
        super().__init__()

    def Stop(self):
        self.flagStop = True

    def beep(self):
        self._beep = True

    def txsensordata(self, strdata):
        self.strdata = strdata
        self.flag_sensorlimit_tx = True

    @Slot()
    def run(self):
        in_waiting = ''
        jMessage = ""
        unit = ''
        itm = ''
        while 1:
            if self.flagStop:
                break
            try:
                in_waiting = self.serialport.in_waiting
            except Exception as e:
                print('Ex:0X07 : ' + str(e))

            while in_waiting == 0:
                time.sleep(0.01)
                try:
                    in_waiting = self.serialport.in_waiting
                except Exception as e:
                    print('Ex:0x08 : ' + str(e))
            try:
                unit = self.serialport.read(in_waiting)
            except Exception as e:
                print('Ex in sensor Thread readline() 527 : ' + str(e))

            if len(unit) > 0:
                try:
                    itm += unit.decode('ascii')
                except:
                    pass

            if b'\n' in unit:
                jMessage = itm  # .decode('ascii')
                itm = ''
                # jMessage += ',' + str(time.perf_counter())
                self.plst = jMessage.split(",")
                self.signal.emit(jMessage)
                if self.pressureque.qsize() <= 0:
                    self.pressureque.put(self.plst[0])

            if self.flag_sensorlimit_tx:
                self.flag_sensorlimit_tx = False
                self.serialport.write(self.strdata.encode('utf-8'))
                time.sleep(0.5)
