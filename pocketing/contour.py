"""
contour.py
---------------

Contour- parallel pocketing strategies and utilities.
"""

from . import graph
from . import polygons


def contour_parallel(polygon, step):
    """
    Create a linked contour parallel milling strategy.
    """
    #
    g, offsets = polygons.offset_graph(
        polygon, distance=step)
    traversal = graph.traverse_child_first(
        g, polygons.closest_node)
    paths = graph.graph_to_paths(
        g, traversal, offsets)

    return paths
