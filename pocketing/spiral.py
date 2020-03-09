"""
spiral.py
------------

Create helical plunges and spirals.
"""
import numpy as np
import networkx as nx

from . import graph
from . import polygons


def offset(polygon, step):
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
        curve = polygons.interpolate(a, b, start=start)
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


def archimedian(
        radius_start,
        radius_end,
        step,
        angular_resolution=None):
    """
    Construct an analytically evaluated Archimedian Spiral
    from radius and per-revolution step.

    The equation of an Archimedian Spiral is:
      r = K * theta

    Parameters
    ------------
    radius_start : float
      Starting radius of the spiral
    radius_end : float
      Maximum radius of the spiral
    step : float
      The distance to step between each full revolution
    angular_resolution : float or None
      The angular resolution of each segment

    Returns
    ------------
    points : (n, 2) float
      Points representing an Archimedian Spiral
    """
    # the spiral constant
    # evaluated from: step = K * 2 * pi
    K = step / (np.pi * 2)

    # use our constant to find angular start and end
    theta_start = radius_start / K
    theta_end = radius_end / K

    # if not passed set angular resolution
    if angular_resolution is None:
        angular_resolution = np.radians(1.0)
    # create an array of angles
    theta = np.arange(
        theta_start, theta_end, step=angular_resolution)
    # use the spiral equation to generate radii
    radii = theta * K
    # convert polar coordinates to 2D cartesian
    points = np.column_stack(
        (np.cos(theta), np.sin(theta))) * radii.reshape((-1, 1))

    return points
