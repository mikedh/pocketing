import trimesh
import pocketing

import matplotlib.pyplot as plt

if __name__ == '__main__':

    path = trimesh.load('../models/wrench.dxf')
    poly = path.polygons_full[0]

    # generate tool paths
    toolpaths = pocketing.contour.contour_parallel(poly, .05)

    # visualize by plotting
    for tool in toolpaths:
        plt.plot(*tool.T)
    path.show()

    
