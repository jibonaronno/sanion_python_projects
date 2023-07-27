# Read SBCR CB dat file, Show the events in Matplotlib subplot graphs, Converted to Little-Endian, Convert To CSV File.

### 1. Input File name is hardcoded. Endian conversion is happened through unpack function: 
integers2 = unpack('<' + 'h' * (len(barray) // 2), barray) . <br>

Format string '<' determines the Endiannes to be Little-endian. 'h' stands for signed short (16 bit). 'h' is multiplied to the number of the size/2 of the byte array barray. so it will return a list of 16 Bit numbers from the byte array.

But I could not save the data after Little-endian conversion to binary file yet. CSV file is created. <br>
getHeader() function converts the endianness of the first 98 byte header data. 