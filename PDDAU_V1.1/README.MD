# PDDAU_V1.1 Virtual PDDAU Python Implementation

... <br>
... <br>

## Act as a virtual PDDAU over local network.

# Requirement
Listen to port 5000 on a PDDAU compatible TCP server. After connection wait for 
commands. 

## Operations
I was facing problems to implement the Layouts in the Main window. The problem 
occurred for a custom widget Mimic. So I place a QWidget in the QT Designer and 
pass this QWidget as a parent for the custom widget Mimic. Mimic is created in 
the main.py class MainWindow construction. 

Mimic is a custom QWidget. At the construction of Mimic it shows an image with 
a rect border around it. The rect border geometry information actually comes from
a json file. 

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
