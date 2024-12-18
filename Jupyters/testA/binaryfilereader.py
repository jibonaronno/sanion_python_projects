'''
 Understand python's struct module for unpack functions.
 In the unpack function
 # '<' indicates the source data is Little Endian
 # 'h' asks for short integer
'''

from pathlib import Path
from struct import unpack

class BinaryFileReader(object):
    def __init__(self):
        self.ficontent = None

    def readFile(self, finame):
        with open(finame, mode='rb') as file:  # b is important -> binary
            self.ficontent = file.read()

    def readFil(self, fullpath):
        filname = fullpath
        print("File name: ", filname)
        self.ficontent = Path(filname).read_bytes()  # Python 3.5+

    def printContent_S8(self, _offset=0):
        barray = self.ficontent[_offset:]
        print("Length : ", len(barray))
        #integers = struct.unpack("!%sH" % (len(barray) // 2), barray)
        integers = unpack('<'+'b'*(len(barray)), barray) # '<' indicates the source data is Little Endian
                                                         # 'b' asks for signed 8bit integer
        print("Type of each element : ", type(integers[0]))
        print(integers)
        pass

    def getArray(self, array_start=-1, array_end=-1):
        # barray = self.ficontent[0:]
        if array_end > 0 and array_start > 0 and array_end > array_start:
            barray = self.ficontent[array_start:array_end]
            integers = unpack('<' + 'b' * (array_end - array_start), barray)  # '<' indicates the source data is Little Endian
            return integers                                    # 'b' asks for signed 8bit integer
        barray = self.ficontent[0:]
        integers = unpack('<' + 'b' * (len(barray)), barray)    # '<' indicates the source data is Little Endian
                                                                # 'b' asks for signed 8bit integer
        return integers

    def printFilContentSize(self, fullpath:str):
        #if self.ficontent != None:
        #    if len(self.ficontent) > 0:
        #        self.ficontent.clear()
        self.readFil(fullpath)
        print(str(len(self.ficontent)))

    def printContent_16(self, _offset=98):
        barray = self.ficontent[_offset:]
        print("Length : ", len(barray))
        integers = unpack('<'+'h'*(len(barray)//2), barray) # '<' indicates the source data is Little Endian
                                                            # 'h' asks for short integer
        print("Type of each element : ", type(integers[0]))
        print(integers)

        # integers = struct.unpack("!%sH" % (len(barray) // 2), barray)

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