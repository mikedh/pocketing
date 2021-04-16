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
from shapely.geometry import LineString, Polygon


from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline, interp1d

from . import graph


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
    Find the distances along a path that result in an set of circles
    that are inscribed to a specified polygon and that have an
    advancing front spaced with a specified step apart.

    Arguments
    -----------
    path : (n, 2) float
      2D path inside a polygon
    polygon : shapely.geometry.Polygon
      Object which contains all of path
    step : float
      How far apart should the advancing fronts of the circles be

    Returns
    -----------
    distance_result : (m) float
      Distances along curve which result in
      nicely spaced circles.
    """
    path = np.asanyarray(path)
    assert trimesh.util.is_shape(path, (-1, 2))
    assert isinstance(polygon, Polygon)

    sampler = trimesh.path.traversal.PathSample(path)
    path_step = step / 25.0

    distance_initial = np.arange(
        0.0, sampler.length + (path_step / 2.0), path_step)

    offset = sampler.sample(distance_initial)
    radius = boundary_distance(polygon=polygon, points=offset)

    pairs = [(offset[0], radius[0])]
    distance_result = [0]

    for point, r, pd in zip(offset[1:],
                            radius[1:],
                            distance_initial[1:]):
        vector = point - pairs[-1][0]
        front_distance = np.linalg.norm(vector) - pairs[-1][1] + r
        if front_distance >= step:
            pairs.append((point, r))
            distance_result.append(pd)
    return np.array(distance_result)


def swept_trochoid(path,
                   polygon,
                   step,
                   counts_per_rotation=360):
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
    counts_per_rotation : int
      Segments in a rotation of the trochoid

    Returns
    ----------
    curve : (n, 2) path
      Curve inside polygon along path.
    """

    path = np.asanyarray(path, dtype=np.float64)
    assert trimesh.util.is_shape(path, (-1, 2))
    assert isinstance(polygon, Polygon)

    # find distances such that overlap is the same
    # between subsequent trochoid circles
    distances = advancing_front(path, polygon, step)

    # smooth distances into sample
    if len(distances) > 3:
        interpolator = UnivariateSpline(
            np.arange(len(distances)), distances, s=0.001)
    elif len(distances) >= 2:
        interpolator = interp1d(
            np.arange(len(distances)), distances)
    else:
        return np.array([])
    sampler = trimesh.path.traversal.PathSample(path)

    x_interp = np.linspace(
        0.0,
        len(distances) - 1,
        len(distances) * counts_per_rotation)
    # smooth distances using our interpolator
    new_distance = interpolator(x_interp)
    new_distance = np.hstack((
        np.tile(new_distance[0], counts_per_rotation),
        new_distance,
        np.tile(new_distance[-1], counts_per_rotation)))
    new_offset = sampler.sample(new_distance)
    new_theta = np.linspace(
        -np.pi * 2,
        np.pi * 2 * len(distances) + np.pi * 2,
        len(new_distance))
    # find the distance from every point to the polygon boundary
    new_radius = boundary_distance(
        polygon=polygon, points=new_offset)
    # calculate the actual trochoid
    curve = trochoid(theta=new_theta,
                     radius=new_radius,
                     offset=new_offset)

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
    Find the indexes on the first curve of where two curves
    intersect.

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
