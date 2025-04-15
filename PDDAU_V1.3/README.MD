# PDDAU_V1.1 Virtual PDDAU Python Implementation

... <br>
... <br>

## Act as a virtual PDDAU over local network.

# Requirement
Listen to port 5000 on a PDDAU compatible TCP server. After connection wait for 
commands. 

## Operations
The threading technique is changed from previous method. Now we are using 
class ServerThread(QThread): and inside ServerThread we are using pyqtSignal 
for sending signal to MainWindow which is main.py. This ServerThread class is 
inside pddsrvr.py file. QThread class has a method self.isInterruptionRequested()
to check if it receives to stop the tread. Since select.select timeout is 0.5 sec
it can check for interruption at every 0.5 sec. 

As we are using select module for TCP communication, initially created the TCP 
object as nonblocking mode, we applied a timeout parameter value of 0.5 for 
select.select(...) method. So it does not block inside the thread loop. 

select.select(..) method is fed with a list of sockets, this list of sockets 

I was facing problems to implement the Layouts in the Main window. The problem 
occurred for a custom widget Mimic. So I place a QWidget in the QT Designer and 
pass this QWidget as a parent for the custom widget Mimic. Mimic is created in 
the main.py class MainWindow construction. 

Mimic is a custom QWidget. At the construction of Mimic it shows an image with 
a rect border around it. The rect border geometry information actually comes from
a json file. 

# Qt Designer Issue
The official QT Designer Latest version generated UI is not compatible with this 
app or may be not compatible with some packages. I had to use pyqt5_qt5_designer 
in windows from pycharm package manager. In Ubuntu I installed pip install pyqt5-tools 
and start the designer by pyqt5-tools designer.
Here are the packages working versions in windows:
1. PyQt5 5.15.11
2. PyQt5-Qt 5.15.2
3. PyQt5_Sip 12.17.0
4. PyQt5-Qt5 5.15.2
5. pyqt5_qt5_designer 0.0.14


# Ubuntu 22.xx wayland PyQt5 xcb errors sanion_python_projects
Install Setup Tools
sudo apt install python-setuptools
sudo apt install python3.10-distutils

sudo apt install libxcb-cursor0

If Qt5 installed earlier by pycharm then uninstall by following commands
sudo python3 /snap/pycharm-community/439/plugins/python-ce/helpers/packaging_tool.py uninstall PyQt5
do same for PyQt5-Qt5, PyQt5_sip, QtPy, qtmodern, pyqtgraph, 

echo $XDG_SESSION_TYPE # Should be x11 or wayland

sudo nano /etc/gdm3/custom.conf
add
WaylandEnable=false

sudo nano /etc/environment
#### Add QT_QPA_PLATFORM=xcb
#### Or No need to edit, 

sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

```Python
```
