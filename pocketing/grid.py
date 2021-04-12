import trimesh
import numpy as np


def cover(polygon, step):
    """
    Cover a polygon's OBB with a uniform grid.

    Parameters
    -----------
    polygon : shapely.geometry.Polygon
      Polygon to be covered
    step : float
      Distance between grid rows

    Returns
    ------------
    path : (n, 2) float
      Points making up the grid path
    """
    if hasattr(polygon, 'exterior'):
        # polygons
        points = np.array(polygon.exterior.coords)
    elif hasattr(polygon, 'coords'):
        # linestrings etc
        points = np.array(polygon.coords)
    else:
        # numpy arrays
        points = np.array(polygon)

    # find the OBB of the points passed
    rotation, extents = trimesh.bounds.oriented_bounds_2D(points)

    ceil = np.ceil(extents / step) * step
    half = ceil / 2.0

    path = trimesh.transform_points(
        create(bounds=[-half, half], step=step),
        matrix=np.linalg.inv(rotation))

    if False:
        import matplotlib.pyplot as plt
        plt.plot(*path.T, color='r', linestyle='-')
        # plt.scatter(*polygon.T)
        plt.show()

    return path


def create(bounds, step):
    """
    Create a grid inside an axis aligned bounding box.

    Parameters
    ----------
    bounds : (2, 2) float
      Axis aligned 2D bounding box
    step : float
      Distance between grid lines
    """
    bounds = np.array(bounds, dtype=np.float64)

    extents = bounds.ptp(axis=0)

    if extents[0] > extents[1]:
        flip = True
        bounds = np.fliplr(bounds)
    else:
        flip = False

    # make sure we've included the whole boundary
    steps = np.ceil(extents / step).astype(int)
    # linearly space between bounds
    sp = np.linspace(*bounds[:, 0], steps[0])

    # do array wangling
    x = (np.tile(sp, (2, 1)).T).ravel()
    y = (np.ones(len(sp) * 2).reshape((len(sp), 2)) * bounds[:, 1])
    # flip every other column
    y[::2] = np.fliplr(y[::2])
    y = y.ravel()

    if flip:
        result = np.column_stack((y, x))
    else:
        result = np.column_stack((x, y))

    return result
