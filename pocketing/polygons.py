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
from shapely.geometry import Polygon, MultiPoint


def hash(polygon):
    """
    Get a hash of a polygon

    Parameters
    -----------
    polygon : shapely.geometry.Polygon
       Polygon to hash

    Returns
    ------------
    crc : int
        Hash of polygon
    """
    crc = zlib.adler32(polygon.wkb)
    return crc


def boundary_distance(polygon, points):
    """
    Find the distance between a polygon's boundary and an
    array of points.

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
      Polygon to query
    points : (n, 2) float
      2D points

    Returns
    ------------
    distance : (n,) float
      Minimum distance from each point to polygon boundary
    """

    try:
        import pygeos
        # the pygeos way is 5-10x faster
        pg_points = pygeos.points(*points.T)
        pg_boundary = pygeos.boundary(pygeos.Geometry(polygon.wkt))
        distance = pygeos.distance(pg_boundary, pg_points)
    except BaseException:
        # in pure shapely we have to loop
        inverse = polygon.boundary
        distance = np.array([
            inverse.distance(i) for i in MultiPoint(points)])

    return distance


def closest_node(g, node, node_options):
    """
    Find the things.

    Parameters
    ------------
    g : networkx.DiGraph
        Graph
    node : hashable
        Node key in g
    node_options : list
        what the fuck

    Returns
    ------------
    closest : hashable
        Member of node_options closest to origin
    """
    if len(node_options) == 1:
        return node_options[0]

    node_options = list(set(node_options).difference({node}))

    origin = g.node[node]['polygon'].centroid.coords[0]
    others = [g.node[i]['polygon'].centroid.coords[0]
              for i in node_options]

    tree = spatial.cKDTree(others)

    distance, index = tree.query(origin, k=1)
    closest = node_options[index]
    return closest


def offset_graph(polygon,
                 distance,
                 min_area=1e-3):
    """
    Generate a graph from a polygon offset inwards until
    the polygon is consumed.

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
        Source geometry to offset
    distance : float
        Distance each step will be offset by
    min_area : float
        Polygons smaller than this will be
        considered fully consumed

    Returns
    -------------
    graph : networkx.DiGraph
        Topology of offsets
    offsets : dict
        The actual offset geometry for each node
        {node key : shapely.geometry.Polygon}
    """

    # make sure distance is negative
    distance = -np.abs(distance)
    # generate the graph of offset polygons
    g = nx.DiGraph()
    # which polygons need to be visited
    queue = collections.deque([polygon])
    # store the polygons at each offset level
    offsets = {}

    while len(queue) > 0:
        current = queue.pop()
        # use polygon hashes as node names
        current_index = hash(current)

        # store the current polygon
        offsets[current_index] = current

        # offset polygon inwards
        buffered = current.buffer(distance)
        for child in trimesh.util.make_sequence(buffered):
            if child.area < min_area:
                continue
            # replace the childs buffered interior with the
            # interior from the original polygon
            child = Polygon(shell=child.exterior.coords,
                            holes=[i.coords
                                   for i in polygon.interiors])
            g.add_edge(hash(current),
                       hash(child))
            queue.append(child)

    return g, offsets


def interpolate(a, b, start=None, step=.005):
    """
    Interpolate between two polygons

    Parameters
    -------------
    a : shapely.geometry.Polygon
        Polygon start point will lie on
    b : shapely.geometry.Polygon
        Polygon end point will lie on
    start : (2,) float, or None
        Point to start at
    step : float
        How far apart should points on
        the path be.

    Returns
    -------------
    path : (n, 2) float
       Path interpolated between polygon exteriors
    """

    # resample the first polygon so every sample is spaced evenly
    ra = trimesh.path.traversal.resample_path(a.exterior,
                                              step=step)

    if start is not None:
        # find the closest index on polygon 'a'
        # by creating a KDTree
        tree_a = spatial.cKDTree(ra)
        index = tree_a.query(start)[1]
        ra = np.roll(ra, -index, axis=0)

    # resample the second polygon for even spacing
    rb = trimesh.path.traversal.resample_path(b.exterior,
                                              step=step)
    # we want points on 'b' that correspond index- wise
    # the resampled points on 'a'
    tree_b = spatial.cKDTree(rb)
    # points on b with corresponding indexes to ra
    pb = rb[tree_b.query(ra)[1]]

    # linearly interpolate between 'a' and 'b'
    weights = np.linspace(0.0, 1.0, len(ra)).reshape((-1, 1))

    # start on 'a' and end on 'b'
    points = (ra * (1.0 - weights)) + (pb * weights)

    return points
