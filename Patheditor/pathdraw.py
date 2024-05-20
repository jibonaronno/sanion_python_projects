'''
https://stackoverflow.com/questions/37514039/selecting-and-editing-specific-markers-using-matplotlib
There are a few ways for you to do this. I doubt there is an easy way for you to drag the line itself.
To drag the line markers (but without seeing any) just make sure they exist but are transparent.
I'm going to use an example from matplotlib documentation: The Path Editor.
Here is the modified code:
'''

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backend_bases import MouseButton
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# fig, ax = plt.subplots()

pathdata = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    # (Path.CLOSEPOLY, (1.58, -2.57)),
]

codes, verts = zip(*pathdata)
path = Path(verts, codes)
patch = PathPatch(path, facecolor='green', edgecolor='yellow', alpha=0.5)
# ax.add_patch(patch)


class PathInteractor:
    """
    A path editor.
    Press 't' to toggle vertex markers on and off.  When vertex markers are on,
    they can be dragged with the mouse.
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, pathpatch, canvas, fig, axis):
        self.canvas = canvas
        self.axis = axis
        self.axis.add_patch(patch)
        self.fig = fig
        self.pathpatch = pathpatch
        self.pathpatch.set_animated(True)

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        xy = self.pathpatch.get_path().vertices
        xyt = self.pathpatch.get_transform().transform(xy)  # to display coords
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        ind = d.argmin()
        return ind if d[ind] < self.epsilon else None

    def buildpath(self, x, y):
        _path = []
        for i in range(len(x)):
            _path.append((Path.MOVETO, (x[i], y[i])))
        return _path