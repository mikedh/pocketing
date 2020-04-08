"""
spiral.py
------------

Create helical plunges and spirals.
"""
import numpy as np
import networkx as nx

from . import graph
from . import polygons
from . import constants


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


def archimedean(
        radius_start,
        radius_end,
        step,
        close=False,
        angle_start=None,
        arc_res=None):
    """
    Construct an Archimedean Spiral from radius and
    per-revolution step.

    The equation of an Archimedean Spiral is:
      r = K * theta

    Parameters
    ------------
    radius_start : float
      Starting radius of the spiral
    radius_end : float
      Maximum radius of the spiral
    step : float
      The distance to step between each full revolution
    close : bool
      Do a full revolution at the final radius or not
    angle_start : float or None
      Start at an angle offset or not
    arc_res : float or None
      The angular size of each returned arc

    Returns
    ------------
    points : (n, 3, 2) float
      Three-point arcs forming an Archimedean Spiral
    """
    # the spiral constant
    # evaluated from: step = K * 2 * pi
    K = step / (np.pi * 2)

    # use our constant to find angular start and end
    theta_start = radius_start / K
    theta_end = radius_end / K

    # if not passed set angular resolution
    if arc_res is None:
        arc_res = constants.default_arc

    arc_count = int(np.ceil((theta_end - theta_start) / arc_res))
    # given that arcs will share points how many
    # points on the helix do we need
    point_count = (2 * arc_count) + 1

    # create an array of angles
    theta = np.linspace(theta_start, theta_end, point_count)
    # use the spiral equation to generate radii
    radii = theta * K

    # make sure they match
    assert np.isclose(radii[0], radius_start)

    # do offset AFTER radius calculation
    if angle_start is not None:
        theta += (angle_start - theta_start)

    # convert polar coordinates to 2D cartesian
    points = np.column_stack(
        (np.cos(theta), np.sin(theta))) * radii.reshape((-1, 1))

    if close:
        # additional arcs to close
        c_count = int(np.ceil((np.pi * 2) / arc_res))
        # additional points added to points
        cp_count = 2 * c_count

        # the additional angles needed to close
        # note we are cutting off the first point as it is a duplicate
        t_close = np.linspace(theta[-1],
                              theta[-1] + np.pi * 2,
                              cp_count + 1)[1:]
        # additional points to close the arc
        closer = np.column_stack((np.cos(t_close),
                                  np.sin(t_close))) * radii[-1]
        # stack points with closing arc
        points = np.vstack((points, closer))
        # add the additional points to the count
        point_count += cp_count

    # convert sequential points into three point arcs
    arcs = points[arc_index(point_count)]

    if constants.strict:
        # check all arcs to make sure the correspond
        for a, b in zip(arcs[:-1], arcs[1:]):
            assert np.allclose(a[2], b[0])

    return arcs


def helix(
        radius,
        height,
        pitch,
        center=None,
        arc_res=None,
        epsilon=1e-8,
        return_angle=False):
    """
    Create an approximate 3D helix using 3-point circular arcs.

    Parameters
    ------------
    radius : float
      Radius of helix
    height : float
      Overall height of helix
    pitch : float
      How far to advance in Z for every rotation
    arc_res : None or float
      Approximately how many radians should returned arcs span
    epsilon : float
      Threshold for zero

    Returns
    --------------
    arcs : (n, 3, 3) float
      Ordered list of 3D circular arcs
    """
    if np.abs(height) < epsilon:
        arcs = np.array([], dtype=np.float64)
        if return_angle:
            return arcs, 0.0
        return arcs

    # set a default arc size if not passed
    if arc_res is None:
        arc_res = constants.default_arc

    # total angle we're traversing
    angle = (np.pi * 2.0 * height) / pitch

    # how many arc sections will result, making sure to ceil
    arc_count = int(np.ceil(angle / arc_res))

    # given that arcs will share points how many
    # points on the helix do we need
    point_count = (2 * arc_count) + 1
    # we're doing 3-point arcs
    theta = np.linspace(0.0, angle, point_count)

    # Z is linearly ramping for every point
    z = np.linspace(0.0, height, len(theta))

    # convert cylindrical to cartesian
    cartesian = np.column_stack(
        (np.cos(theta), np.sin(theta), z))
    # multiply XY by radius
    cartesian[:, :2] *= radius
    if center is not None:
        cartesian[:, :2] += center

    # now arcs are 3 cartesian points
    arcs = cartesian[arc_index(point_count)]

    if return_angle:
        # return the final angle
        final = theta[-1] % (np.pi * 2)
        return arcs, final

    return arcs


def arc_index(count):
    # example of indexes for situation:
    # arc_count = 3
    # point_count = 7
    # index = [[0,1,2],
    #          [2,3,4],
    #          [4,5,6]]

    count = int(count)
    # use index wangling to generate that from counts
    index = np.arange(count - 1).reshape((-1, 2))
    index = np.column_stack((index, index[:, 1] + 1))
    return index
