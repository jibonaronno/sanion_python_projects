# COLLECT DATA FROM MODBUS TCP CLIENT ACQ AND DISPLAY GRAPHS

### * matplotlib, tkinter is used to show the graphs
        Class DASH is created intended to handle ploting works. 
        But ploting parts are disoriented at 2 classes. I created 
        Matplolib's Line objects initiated globally in the class 
        __init__ first to refresh the plots when new data is assigned
        to the plots. 