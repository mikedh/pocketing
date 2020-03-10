from scipy import spatial
from shapely.geometry import LineString
from trimesh.constants import log

import numpy as np
import trimesh


def smooth_inside(path,
                  polygon,
                  max_smoothing=.25,
                  max_overlap=.05,
                  chunks=None):
    """
    """
    path = np.asanyarray(path)
    if not trimesh.util.is_shape(path, (-1, 2)):
        raise ValueError('input path must be (-1,2)!')

    if isinstance(chunks, int):
        result = np.vstack([smooth_inside(
            path=i,
            polygon=polygon,
            max_smoothing=max_smoothing)
            for i in np.array_split(path, chunks)])
        return result

    polygon_test = polygon.buffer(max_overlap)
    if not polygon_test.contains(LineString(path)):
        raise ValueError('input polygon doesn\'t contain path!')
    for smooth in np.linspace(0.0, max_smoothing, 10)[1:][::-1]:
        path_smooth = trimesh.path.simplify.resample_spline(
            path, smooth=smooth, degree=3)
        if polygon_test.contains(LineString(path_smooth)):
            log.info('Smoothed path inside polygon by %f', smooth)
            return path_smooth
    log.info('Unable to smooth path beyond original')
    return path


def simplify_inside(path, polygon, max_distance=.1):

    path_test = LineString(path)
    polygon_test = polygon.buffer(1e-3)
    if not polygon_test.contains(path_test):
        raise ValueError('input polygon doesn\'t contain path!')

    for simplify in np.linspace(0.0, max_distance, 10)[1:][::-1]:
        path_simple = path_test.simplify(simplify)

        if polygon_test.contains(path_simple):
            log.info('Simplified path inside polygon by %f', simplify)
            return np.array(path_simple.coords)
    log.info('Unable to simply path beyond original')

    return path


def check_path(polygon, path, radius, resolution=.01):
    """
    Check a circle being swept along a path, which is contained in a polygon.
    Calculate the removal rate, and check that it stays within the bounds
    throughout the traversal.

    Arguments
    -----------
    polygon: shapely.geometry.Polygon object, the geometry to be carved
    path:    (n,2) float, path through space
    radius:  float, the radius of the tool being swept
    resolution: float, the length of a square side of the pixel

    Returns
    -----------
    resampled: (m,2) float, path resampled at resolution
    removal:   (m,) float, removal rate at each vertex
    """
    # turn the polygon into a set of 2D points on a grid by rasterizing it
    (grid_offset,
     grid,
     grid_points) = trimesh.path.polygons.rasterize_polygon(polygon,
                                                            pitch=resolution)

    # create a KDtree of pixels inside the polygon
    tree = spatial.cKDTree(grid_points)

    # resample path so each vertex is resolution apart
    resampled = trimesh.path.traversal.resample_path(path,
                                                     step=resolution)

    # boolean flag for whether a grid point has been used by the sweep yet
    unused = np.ones(len(grid_points), dtype=np.bool)

    # an integer for the number of pixels intersected by the first
    first = np.zeros(len(resampled), dtype=np.int)

    for i, hit in enumerate(tree.query_ball_point(resampled,
                                                  r=radius)):
        first[i] = unused[hit].sum()
        unused[hit] = False

    # (pixel count * (resolution ** 2)) / (resolution)
    # simplifies to: pixel count * resolution
    removal = first.astype(np.float64) * resolution

    return removal, resampled


def query_nearest(points_original, points_query):
    """
    Find the nearest point from an original set for each of a query set.

    Arguments
    -----------
    points_original: (n,d) float, points in space
    points_query:    (m,d) float, points in space

    Returns
    -----------
    index: (m,) int, index of closest points_original for each points_query
    """
    tree = spatial.cKDTree(points_original)
    distance, index = tree.query(points_query, k=1)
    return index


def intersection_index(curve_a, curve_b):
    """
    Find the indexes on the first curve of where two curves intersect.

    Arguments
    ----------
    curve_a: (n,2) float, curve on a plane
    curve_b: (m,2) float, curve on a plane

    Returns
    ----------
    indexes: (p) int, indexes of curve_a where it intersects curve_b
    """
    hits = np.array(LineString(curve_a).intersection(LineString(curve_b)))
    indexes = np.hstack(query_nearest(curve_a, hits))

    return indexes
