# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from pyModbusTCP.server import ModbusServer, DataBank
from pyModbusTCP.client import ModbusClient
from pprint import pprint


class SRVR(object):
    def __init__(self):
        #self.srvr = ModbusServer("localhost", 100, no_block=False)
        self.clnt = ModbusClient(host='192.168.10.100', port=100, auto_open=False, debug=False)
        self.__address = 8212
        self.voltwave = []

    def cnnct(self):
        try:
            #self.srvr.start()
            ix = 1
            ix = 0
            self.clnt.open()
            #while True:
            while True:
                regs_1 = self.clnt.read_holding_registers(self.__address, 64)
                for reg in regs_1:
                    self.voltwave.append(str(self.__address + ix) + "," + str(reg) + "\n")
                self.__address = self.__address + 64
                if self.__address > 10452:
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    srvr = SRVR()
    srvr.cnnct()
    pass
