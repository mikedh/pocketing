"""
contour.py
---------------

Contour- parallel pocketing strategies and utilities.
"""
import zlib

import trimesh
import collections
import numpy as np
import networkx as nx

from scipy import spatial
from shapely.geometry import Polygon

from . import graph
from . import polygons

def offset_spiral(polygon, step):
    """
    Create a spiral inside a polygon by offsetting the
    polygon by a step distance and then interpolating.

    Parameters
    -----------
    polygon : shapely.geometry.Polygon object
        Source geometry to create spiral inside of
    step : float
        Distance between each step of the curve

    Returns
    ----------
    spiral : (n, 2) float
        Points representing a spiral path
        contained inside polygon
    """
    g, offset = polygons.offset_graph(polygon, step)

    # make sure offset polygon graph topology supports this
    assert graph.is_1D_graph(g)

    # make sure polygons don't have interiors
    assert all(len(v.interiors) == 0 for v in offset.values())

    # starting point for each section of curve
    start = None
    # store a sequence of (m,2) float point curves
    path = []

    # root node is index 0
    nodes = list(nx.topological_sort(g))
    # loop through nodes in topological order
    for i in range(len(nodes) - 1):
        # get the polygons we're looking at
        a = offset[nodes[i]]
        b = offset[nodes[i + 1]]

        # create the spiral section between a and b
        curve = interpolate(a, b, start=start)
        # make sure the next section starts where this
        # section of the curve ends
        start = curve[-1]
        # store the curve in the main sequence
        path.append(curve)

    # stack to a single (n,2) float list of points
    path = np.vstack(path)

    # check to make sure we didn't jump on any step
    # that would indicate we did something super screwey
    gaps = (np.diff(path, axis=0) ** 2).sum(axis=1).max()
    assert gaps < 1e-3

    return path


def contour_parallel(polygon, step):
    """
    Create a linked contour parallel milling strategy.
    """
    g, offsets = polygons.offset_graph(polygon,
                                      distance=step)
    traversal = graph.traverse_child_first(g,
                                           polygons.closest_node)
    paths = graph.graph_to_paths(g,
                                 traversal,
                                 offsets)

    return paths

