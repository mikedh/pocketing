import trimesh
import pocketing

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load 2D geometry into polygon object
    path = trimesh.load('../models/wrench.dxf')
    polygon = path.polygons_full[0]
    
    radius = .125
    step = radius * 0.75

    toolpath = pocketing.trochoidal.toolpath(polygon,
                                              step=step)


    plt.plot(*toolpath.T)
    path.show()
    
