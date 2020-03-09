"""
visualize.py
---------------

Utilities for visualizing 2D toolpaths
"""
import numpy as np


def animate_path(polygon, paths):
    """
    Animate a path being swept inside a polygon
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # initialization function: plot the background of each frame
    def init():
        ax.plot(*polygon.exterior.xy)
        for i in polygon.interiors:
            ax.plot(*i.xy)
        [i.set_data([], []) for i in lines]

        return []

    # animation function.  This is called sequentially
    def animate(i):
        current_line = np.searchsorted(paths_cs, i)
        current_index = i - np.append(0, paths_cs)[current_line]

        lines[current_line].set_data(*paths[current_line][:current_index].T)
        return []

    bounds = (np.array(polygon.bounds) + .25 *
              np.array([-1, -1, 1, 1])).reshape((2, 2))

    fig = plt.figure()
    ax = plt.axes(xlim=bounds[:, 0], ylim=bounds[:, 1])
    # ax.set_aspect('equal', 'datalim')
    # line, = ax.plot([], [], lw=2)
    # circ, = ax.plot([], [], lw=1, color='r')

    lines = [ax.plot([], [], lw=2)[0] for i in range(len(paths))]

    paths_len = [len(i) for i in paths]
    paths_cs = np.cumsum(paths_len)

    animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=paths_cs[-1],
        interval=.25,
        blit=False)
    plt.show()
