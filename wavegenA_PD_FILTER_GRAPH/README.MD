# WAVEGEN Function Generator

... <br>
... <br>

## PD Data Filter Graph from Flowmeter_PD_FILTER_GRAPH and wavegenA_PD_FILTER_GRAPH <br>

### Final code will be merged into wavegenA_PD_FILTER_GRAPH. <br>
The wavegenA project was organised for reading PD file or any binary data file and show 
the graph. Present tool (wavegenA_PD_FILTER_GRAPH) will show or animate the PD data and 
Filtered PD Data. Practically 2 graph top bottom at beggining stage. The main wavegenA 
project can Initiate a File Open Dialog and show a window (mimic.py) which is only 3 tab 
window where 2nd tab contains the QGraphicsView Container. <br>

This version allows user to select a folder and it lists all the files of the folder in a 
list view. After selecting a folder a dictionary 'self.file_path_dictionary' kept all the files 
path as value with file names as keys. 
<br>
After select a folder the Treeview at left side will show the list of files. User can select 
any file. The file processing takes about 10 to 15 second. Then it will show the original original 
signal and the filtered signal. <br>
<img alt="NO IMAGE" src="img/img051.jpg"><br><br>


```Python
```
