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
        center=None,
        close=False,
        point_start=None,
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

    if radius_start > radius_end:
        sign = 1
    else:
        sign = -1

    # the spiral constant
    # evaluated from: step = K * 2 * pi
    K = step / (np.pi * 2)

    # use our constant to find angular start and end
    theta_start = radius_start / K
    theta_end = radius_end / K

    # if not passed set angular resolution
    if arc_res is None:
        arc_res = constants.default_arc

    arc_count = int(np.ceil((
        abs(theta_end - theta_start)) / arc_res))

    # given that arcs will share points how many
    # points on the helix do we need
    arc_index, point_count = arc_indexes(arc_count)

    assert arc_index.max() == point_count - 1

    # create an array of angles
    theta = np.linspace(theta_start, theta_end, point_count)

    # use the spiral equation to generate radii
    radii = theta * K

    # make sure they match
    assert np.isclose(radii[0], radius_start)
    assert np.isclose(radii[-1], radius_end)

    # do offset AFTER radius calculation
    if angle_start is not None:
        theta += (angle_start - theta_start)

    # convert polar coordinates to 2D cartesian
    points = np.column_stack(
        (np.cos(theta), np.sin(theta))) * radii.reshape((-1, 1))

    if close:

        # get indexes of arcs required to close
        close_idx, close_ct = arc_indexes(
            int(np.ceil((np.pi * 2) / arc_res)))

        # the additional angles needed to close
        # we are cutting off the first point as its a duplicate
        t_close = np.linspace(theta[-1],
                              theta[-1] + np.pi * 2 * sign,
                              close_ct)[1:]

        # additional points to close the arc
        closer = np.column_stack((
            np.cos(t_close), np.sin(t_close))) * radii[-1]
        assert len(closer) == close_ct - 1
        assert len(points) == point_count

        # stack points with closing arc
        points = np.vstack((points, closer))
        # add the additional points to the count
        point_count += close_ct - 1
        # add the additional arc indexes

        arc_index = np.vstack((
            arc_index, arc_index[-1][-1] + close_idx))

        assert len(points) == point_count
        # max index of arcs should correspond to points
        assert arc_index[-1][-1] == point_count - 1

    if center is not None:
        points += center

    # convert sequential points into three point arcs
    arcs = points[arc_index]

    if constants.strict:
        # check all arcs to make sure the correspond
        for a, b in zip(arcs[:-1], arcs[1:]):
            assert np.allclose(a[2], b[0])

    if point_start is not None:
        a, b = np.clip(
            (point_start[:2] - center[:2]) / radius_start,
            -1.0, 1.0)
        assert np.isclose(a, np.cos(angle_start), atol=1e-3)
        assert np.isclose(b, np.sin(angle_start), atol=1e-3)

    return arcs


def helix(radius,
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

    arc_index, point_count = arc_indexes(arc_count)

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
    else:
        center = [0, 0, 0]
    # now arcs are 3 cartesian points
    arcs = cartesian[arc_index]

    if return_angle:
        # return the final angle
        helix_end = theta[-1] % (np.pi * 2)

        vec = arcs[-1][-1][:2] - center[:2]
        # norm of arc should be close to radius
        assert np.isclose(np.linalg.norm(vec), radius, rtol=1e-3)
        # check to make sure the angle is accurate
        a, b = np.clip(vec / radius, -1.0, 1.0)
        ac, bc = np.cos(helix_end), np.sin(helix_end)

        assert np.isclose(a, ac, atol=1e-3)
        assert np.isclose(b, bc, atol=1e-3)

        return arcs, helix_end

    return arcs


def arc_indexes(arc_count):
    """
    Parameters
    -----------
    count : int
      Number of arcs

    Returns
    ----------
    arc_index : (n, 3) int
      Indexes of arcs sharing common points
    points_count : int
      Number of range points

    Examples
    ----------
    0  1  2  3  4  5  6
    *--*--x--*--x--*--*
    [[0,1,2],[2,3,4],[4,5,6]]
    arc_count:3, point_count=7

    0  1  2
    *--*--*
    [[0,1,2]] 3
    arc_count:1 point_count=3

    0  1  2  3  4  5  6  7  8  9  10 11 12
    *--*--x--*--x--*--x--*--x--*--x--*--x
    [[0,1,2],[2,3,4],[4,5,6],[6,7,8],[8,9,10],[10,11,12]]
    arc count: 6, point_count: 13
    """

    # given that arcs will share points how many
    # points on the helix do we need
    point_count = (2 * arc_count) + 1

    # use index wangling to generate that from counts
    arc_index = np.arange(point_count - 1).reshape((-1, 2))
    arc_index = np.column_stack((
        arc_index, arc_index[:, 1] + 1))

    assert arc_index[-1][-1] == point_count - 1

    assert arc_index.max() == (point_count - 1)

    return arc_index, point_count
