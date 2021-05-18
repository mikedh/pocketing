"""
trochoidal.py
------------------

Generate troichoidal toolpaths or a bunch of tiny
little circle-ish shapes, generally used for high
speed milling as you can execute it with continuous
high accelerations and it has good chip-clearing.
"""

import trimesh
import numpy as np

from .polygons import boundary_distance
from shapely.geometry import LineString, Polygon, Point


from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline, interp1d

from . import graph
from . import spline
import matplotlib.pyplot as plt


def constrained_spiral(a, b, r, counts=200, kind=None):
    """
    Create a 360 degree spiral that starts and ends
    at defined points and is always constrained to
    be within `r` of the line between `a` and `b`.
    """

    # find the vector from start to end
    vec = b - a
    # the total length we're traversing
    norm = np.linalg.norm(vec)
    # the signed angle of the vector
    angle = np.arctan2(*(vec / norm)[::-1])

    theta = np.linspace(0, np.pi * 2, counts)
    linear = np.linspace(0, norm, counts)
    x = r * -np.cos(theta) + linear + r
    y = r * np.sin(theta)

    pts = np.column_stack((x, y))

    eff = np.linalg.norm(pts - np.column_stack(
        (linear, np.zeros(counts))), axis=1)

    pts /= (0.3 * (eff / r) + 1).reshape((-1, 1))

    #contain = LineString([a, b]).buffer(r)
    contain = LineString([[0, 0], [norm, 0]]).buffer(r)
    plt.plot(*contain.exterior.xy)

    plt.plot(*pts.T)
    plt.show()

    from IPython import embed
    embed()

    # needs to start at the origin
    # assert np.isclose(x[0], 0.0)
    # needs to end at correct vector length
    # assert np.isclose(x[-1], norm)
    # curve should not exceed boundary
    assert x.max() < (norm + r + 1e-4)
    # should start and end touching X axis
    assert np.isclose(y[0], 0.0)
    assert np.isclose(y[-1], 0.0)

    hand = np.cross(np.append(vec, 0), [0, 1, 0])
    #hand /= np.linalg.norm(hand)
    # print(hand)

    # if hand[-1] < 0.0:
    #    troch[:,1] *= -1

    rot = trimesh.transformations.rotation_matrix(
        angle, [0, 0, 1])[:2, :2]
    points = np.dot(rot, troch.T).T + a


def trochoid(offset, theta, radius):
    """
    Produce a raw unfiltered trochoid.

    Parameters
    ------------
    offset : (n, 2) float
      Cartesian offset for each position
    theta : (n,) float
      Angle in radians for each step
    radius : (n,) float
      Radius at each step

    Returns
    -----------
    troch : (m, 2) float
      Trochoidal path as polyline.
    """
    x = offset[:, 0] + radius * np.cos(theta)
    y = offset[:, 1] + radius * np.sin(theta)
    troch = np.column_stack((x, y))

    return troch


def advancing_front(path, polygon, step):
    """
    Find the distances along a path that result in circles
    that are inscribed to a specified polygon and that have an
    advancing front spaced with a specified step apart.

    Parameters
    -----------
    path : (n, 2) float
      2D path inside a polygon
    polygon : shapely.geometry.Polygon
      Object which contains all of path
    step : float
      How far apart should the advancing fronts of the
      circles be.

    Returns
    -----------
    distance_result : (m) float
      Distances along curve which result in
      nicely spaced circles.
    """
    path = np.asanyarray(path)
    assert trimesh.util.is_shape(path, (-1, 2))
    assert isinstance(polygon, Polygon)

    # so we can get points along the path
    sampler = trimesh.path.traversal.PathSample(path)
    path_step = step / 25.0

    # get a first sample of ditances along the path
    initial = np.arange(
        0.0, sampler.length + (path_step / 2.0), path_step)
    # sample points on the path
    offset = sampler.sample(initial)
    # find the radii at these initial samples
    radius = boundary_distance(polygon=polygon, points=offset)

    # now loop through and collect new points
    # sampled vertices
    points = [offset[0]]
    # radii at these vertices
    radii = [radius[0]]
    result = [0]

    for point, r, pd in zip(
            offset[1:],
            radius[1:],
            initial[1:]):
        vector = point - points[-1]
        norm = np.linalg.norm(vector)
        front_distance = norm - radii[-1] + r
        if front_distance >= step:
            radii.append(r)
            points.append(point)
            result.append(pd)

    return np.array(result)


def swept_trochoid(path,
                   polygon,
                   step,
                   smooth_factor=2,
                   counts=200):
    """
    Generate a swept trochoid along a path with the following
    properties:
      1) contained inside polygon
      2) fronts of trochoid are separated by step distance
      3) divided into approximately counts_per_rotation
         for each rotation

    Parameters
    -------------
    path : (n, 2) float
      Path in 2D to generate trochoid along
    polygon : shapely.geometry.Polygon
      Object that will contain result
    step : float
      Distance between subsequent rotations of the trochoid.
    counts : int
      Segments in a rotation of the trochoid

    Returns
    ----------
    curve : (n, 2) path
      Curve inside polygon along path.
    """

    path = np.asanyarray(path, dtype=np.float64)
    assert trimesh.util.is_shape(path, (-1, 2))
    assert isinstance(polygon, Polygon)

    # find distances such that overlap is the
    # same between subsequent trochoid circles
    distances = advancing_front(path, polygon, step)

    # all piecewise trochoids must start and end on the path
    cutter = LineString(path)
    # given a distance return the cartesian points
    sampler = trimesh.path.traversal.PathSample(path)

    on_path = sampler.sample(distances)

    control = []

    for a, b in zip(on_path[:-1], on_path[1:]):

        # linearly interpolate between start and end points
        # rather than exactly follow the path between points
        samples = np.linspace(
            0, 1, 30).reshape((-1, 1)) * (b - a) + a
        assert np.allclose(samples[0], a)
        assert np.allclose(samples[-1], b)
        # find the smallest radius sample
        # rather than use variable radius
        radii = boundary_distance(
            polygon=polygon, points=samples)

        troch = spline.section(points=np.array(
            [a, b]), radius=radii[-1], debug=False)

        plt.plot(*Point(a).buffer(radii[0]).exterior.xy,
                 linestyle='dashed', color='g')
        plt.plot(*Point(b).buffer(radii[-1]).exterior.xy,
                 linestyle='dashed', color='g')

        # troch  constrained_spiral(a, b, r=radii.min())

        control.append(spline.discretize_bezier(troch))

    plt.plot(*np.vstack(control).T)
    trimesh.path.polygons.plot(polygon)
    from IPython import embed
    embed()

    return curve


def query_nearest(points_original, points_query):
    """
    Find the nearest point from an original set for each of a
    query set.

    Arguments
    -----------
    points_original : (n, d) float
      Points in space
    points_query : (m, d) float
      Points in space

    Returns
    -----------
    index : (m,) int
      Index of closest points_original for each points_query
    """
    tree = cKDTree(points_original)
    distance, index = tree.query(points_query, k=1)
    return index


def intersection_index(curve_a, curve_b):
    """
    Find the indexes on the first curve of where two
    curves intersect.

    Arguments
    ------------
    curve_a : (n, 2) float
      Curve on a plane
    curve_b : (m, 2) float
      Curve on a plane

    Returns
    ----------
    indexes : (p) int
      Indexes of curve_a where it intersects curve_b
    """
    hits = np.array(LineString(curve_a).intersection(
        LineString(curve_b)))
    indexes = np.hstack(query_nearest(curve_a, hits))

    return indexes


def toolpath(polygon,
             step,
             start_point=None,
             start_radius=None,
             medial=None,
             min_radius=None):
    """
    Calculate a trochoidal (bunch of little circles) toolpath
    for a given polygon with a tool radius and step.

    Parameters
    --------------
    polygon : shapely.geometry.Polygon
      Closed region to fill with tool path
    step : float
      Distance to step over between cuts
    min_radius : None, or float
      Minimum radius toolpaths are allowed to be

    Returns
    ---------------
    paths : sequence of (n, 2) float
      Cutting tool paths
    """

    if polygon is None or polygon.area < 1e-3:
        raise ValueError('zero area polygon!')
    # if not specified set to fraction of stepover
    if min_radius is None:
        min_radius = step / 2.0

    # resolution for medial axis calculation
    resolution = np.diff(np.reshape(polygon.bounds, (2, 2)),
                         axis=0).max() / 500.0
    # the skeleton of a region
    if medial is None:
        medial = trimesh.path.Path2D(
            **trimesh.path.exchange.misc.edges_to_path(
                *trimesh.path.polygons.medial_axis(
                    polygon,
                    resolution=resolution)))
    # find the radius of every medial vertex
    medial_radii = boundary_distance(
        polygon=polygon,
        points=medial.vertices)

    g = medial.vertex_graph
    # remove nodes below the minimum radius
    g.remove_nodes_from(np.nonzero(
        medial_radii < min_radius)[0])

    if start_point is None:
        # if no passed start point use the largest radius
        start = medial_radii.argmax()
    else:
        # start from the vertex closest to the passed point
        start = query_nearest(
            medial.vertices, start_point)

    # a flat traversal which visits every node
    # and where every consecutive value is an edge
    order = graph.dfs(g, start=start)
    # area where we're cutting
    cut = swept_trochoid(
        path=medial.vertices[order],
        polygon=polygon,
        step=step)

    return cut
