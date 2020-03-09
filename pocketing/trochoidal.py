"""
trochoidal.py
------------------

Generate troichoidal toolpaths, or a bunch of tiny little circles.
Generally used for high speed milling.
"""

import trimesh
import numpy as np
import networkx as nx

from .polygons import boundary_distance

from shapely.geometry import LineString, Polygon

from collections import deque
from scipy.interpolate import UnivariateSpline, interp1d
from scipy import spatial


def trochoid(offset, theta, radius):
    """
    Produce a raw unfiltered trochoid

    Parameters
    ------------
    offset : (n,2) float
        Cartesian offset for each position
    theta : (n,) float
        Angle in radians for each step
    radius : (n,) float
        Radius at each step

    Returns
    -----------
    troch : (m,2) float
        Trochoidal path
    """
    x = offset[:, 0] + radius * np.cos(theta)
    y = offset[:, 1] + radius * np.sin(theta)
    troch = np.column_stack((x, y))

    return troch


def traversal(polygon, min_radius=.06):
    """
    Return a traversal of a polygon along the medial axis.

    Arguments
    ------------
    polygon : shapely.geometry.Polygon object
    min_radius : float, minimum radius to go

    Returns
    ------------
    slow : (n, 2) float
      Traversal of nodes that have never been seen before
    fast : (n, 2) float
      Traversal of previously visited nodes
    """
    resolution = np.diff(np.reshape(
        polygon.bounds, (2, 2)), axis=0).max() / 200.0
    medial_e, medial_v = trimesh.path.polygons.medial_axis(
        polygon,
        resolution=resolution)

    medial = trimesh.path.Path2D(
        **trimesh.path.exchange.misc.edges_to_path(
            medial_e, medial_v))

    medial_radii = boundary_distance(
        polygon=polygon, points=medial_v)

    g = medial.vertex_graph

    bad = np.nonzero(medial_radii < min_radius)[0]
    g.remove_nodes_from(bad)

    start = query_nearest(
        medial.vertices, polygon.centroid.coords)
    current = deque([start])
    slow = deque()
    fast = deque()

    for e in nx.dfs_edges(g, start):
        if current[-1] != e[0]:
            closest = current[-1]
            shortest = nx.shortest_path(g, closest, e[0])
            fast.append(shortest)
            slow.append(np.array(current))
            current = deque([e[0]])
        current.append(e[1])
    slow.append(current)

    slow = [medial.vertices[i] for i in slow]
    fast = [medial.vertices[i] for i in fast]
    return slow, fast


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

    pairs = deque([(offset[0], radius[0])])

    distance_result = deque([0])

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
      3) divided into approximatly counts_per_rotation
         for each rotation

    Parameters
    -------------
    path : (n, 2) float
      Path in 2D to generate trochoid along
    polygon : shapely.geometry.Polygon
      Object that will contain result
    step : float
      Distance between subsequent rotations of the trochoid
    counts_per_rotation : int
      Segments in a rotation of the trochoid

    Returns
    ----------
    curve : (n, 2) path
      Toolpath
    """

    path = np.asanyarray(path)
    assert trimesh.util.is_shape(path, (-1, 2))
    assert isinstance(polygon, Polygon)

    # path = toolpath.smooth_inside(path,
    #                              polygon,
    #                              max_smoothing=.25,
    #                              max_overlap=.05)

    distances = advancing_front(path, polygon, step)

    if len(distances) > 3:
        interpolator = UnivariateSpline(np.arange(len(distances)),
                                        distances,
                                        s=.01)
    elif len(distances) >= 2:
        interpolator = interp1d(np.arange(len(distances)),
                                distances)
    else:
        return []

    x_interp = np.linspace(0.0,
                           len(distances) - 1,
                           len(distances) * counts_per_rotation)

    sampler = trimesh.path.traversal.PathSample(path)

    new_distance = interpolator(x_interp)
    new_distance = np.hstack((np.tile(new_distance[0],
                                      counts_per_rotation),
                              new_distance,
                              np.tile(new_distance[-1],
                                      counts_per_rotation)))
    new_offset = sampler.sample(new_distance)
    new_theta = np.linspace(-np.pi * 2,
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
    tree = spatial.cKDTree(points_original)
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
             min_radius=None):
    """
    Calculate a troichoidal (bunch of little circles) toolpath
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

    # if not specified set to fraction of stepover
    if min_radius is None:
        min_radius = step / 2.0

    # resolution for medial axis calculation
    resolution = np.diff(np.reshape(polygon.bounds,
                                    (2, 2)), axis=0).max() / 500.0
    # the skeleton of a region
    medial_e, medial_v = trimesh.path.polygons.medial_axis(
        polygon,
        resolution=resolution)
    medial = trimesh.path.Path2D(
        **trimesh.path.exchange.misc.edges_to_path(
            medial_e, medial_v))

    medial_radii = boundary_distance(polygon=polygon,
                                     points=medial_v)

    g = medial.vertex_graph

    bad = np.nonzero(medial_radii < min_radius)[0]
    g.remove_nodes_from(bad)

    # start from the medial vertex closest to the centroid
    start = query_nearest(medial.vertices,
                          polygon.centroid.coords)[0]
    current = deque([start])

    slow = deque()
    fast = deque()

    for edge in nx.dfs_edges(g, start):
        # check to see if we have jumped a
        if current[-1] != edge[0]:
            """
            Case where we have jumped between two nodes which
            aren't connected.
            """
            jump = nx.shortest_path(g, current[-1], edge[1])
            jump_length = LineString(medial.vertices[jump]).length
            radii = medial_radii[[current[-1], edge[1]]]
            jump_ratio = jump_length / radii.max()

            if jump_ratio > 1.0:
                fast.append(jump)
                slow.append(np.array(current))
                current = deque([edge[0]])
        current.append(edge[1])
    slow.append(current)

    path = deque()
    for index in range(len(slow)):
        vertices = medial.vertices[slow[index]]

        path.extend(swept_trochoid(path=vertices,
                                   polygon=polygon,
                                   step=step))

        if index < len(fast):
            path.extend(medial.vertices[fast[index]])

    path = np.array(path)
    return path
