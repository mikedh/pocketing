"""
spline.py
------------

Produce a single spline section that removes a
specified radius.
"""

import trimesh
import numpy as np

from .resources import get_path

from trimesh.path.curve import discretize_bezier


def gold_to_control(file_name):
    m = trimesh.load(file_name)
    m.vertices[:, 1] *= -1.0

    pt = m.vertices[[m.entities[0].points[0],
                     m.entities[-1].points[-1]]]

    assert np.isclose(*pt[:, 0])

    a = m.vertices[m.entities[0].points[0]]
    m.apply_translation(-a)

    ai = m.entities[0].points[0]
    bi = m.entities[-1].points[-1]
    a = m.vertices[ai]
    b = m.vertices[bi]

    # we're just at radius
    r = (m.bounds[1] - b)[1]

    import networkx as nx
    p = nx.shortest_path(m.vertex_graph, ai, bi)

    return m.vertices[p]


control = gold_to_control(get_path('gold.svg'))


def section(radius,
            points, debug=False):
    """
    Produce the control points for a cut section that
    starts and ends at specified points.

    Parameters
    -----------
    radius : float
      Radius of slot to cut.
    points : (2, 2) float
      Start and end point for the slot.

    Returns
    ----------
    control : (n, 2) float
      Control points for a bezier curve.
    """

    angle_start = 0.08726646259971647
    angle_end = 2.6179938779914944

    assert points.shape == (2, 2)
    f_vector = points[1] - points[0]
    length = np.linalg.norm(f_vector)
    discrete = discretize_bezier(control)

    # angle of control points
    ang = np.arctan2(*(control - control[-1]).T[::-1])

    # which angles are in the range that should be on the circle
    ang_ok = (ang > angle_start) & (ang < angle_end)

    # find the angles of every discretized point
    ang_dis = np.arctan2(*(discrete - discrete[-1]).T[::-1])

    # find the indexes of discrete for each control point
    idx_dis = np.searchsorted(ang_dis, ang[ang_ok])

    vec_dis = discrete[idx_dis] - discrete[-1]
    rad_dis = np.linalg.norm(vec_dis, axis=1)

    scaling = radius / (1.001 * discrete[:, 0].max())
    c_new = control.copy() * scaling
    c_new[ang_ok] = (control[ang_ok] - control[-1]) * \
        (radius / rad_dis).reshape((-1, 1)) + c_new[-1]

    Y = c_new[:, 1]
    length_current = Y[-1]
    move = length_current - length

    # we are scaling anything behind end point
    index = Y < length_current  # (length_current - radius/3)

    # scale control points behind circle-following
    scaling = (length_current - Y[index]) / length_current
    c_new[:, 1][index] += scaling * move
    c_new[:, 1] -= move

    assert np.allclose(c_new[0], 0.0)
    assert np.allclose(c_new[-1], [0.0, length])

    # get the angle the requested final vector is along
    f_angle = np.arctan2(*f_vector[::-1])
    # generate a rotation matrix to align us
    f_rot = trimesh.transformations.planar_matrix(
        points[0], theta=(np.pi / 2) - f_angle)
    # transform control points to specified location
    c_final = trimesh.transform_points(c_new, f_rot)

    # make sure we didn't screw up endpoints
    assert np.allclose(c_final[[0, -1]], points)

    if debug:
        import matplotlib.pyplot as plt
        from shapely.geometry import LineString
        plt.plot(*LineString(points).buffer(radius).exterior.xy,
                 linestyle='dashed', color='k')
        plt.scatter(*points.T)
        plt.plot(*discretize_bezier(c_final).T, color='b')
        plt.show()

    return c_final


if __name__ == '__main__':
    trimesh.util.attach_to_log()
    #m = trimesh.load('spiral.svg')
    #m = trimesh.load('spline_path.DXF')

    length = 2.0
    radius = 1.0

    cnt = refine(
        radius=radius,
        points=np.array(
            [[1.2, 1.0], [3, 0.5]]))
    m = trimesh.path.Path2D(
        entities=[trimesh.path.entities.Bezier(
            points=np.arange(len(cnt)))],
        vertices=cnt,
        process=False)

    """
    epsilon = 1e-3

    #X, Y = #m.vertices.T
    X, Y = cnt.T
    length_current = Y[-1]
    move = length_current - length

    # we are scaling anything behind end point
    index = Y < length_current #(length_current - radius/3)

    # scale control points behind circle-following
    scaling = (length_current - Y[index]) / length_current
    m.vertices[:,1][index] += scaling * move
    m.vertices[:,1] -= move

    #m.vertices[:,0][index] -= X[index] * (scaling / radius)
    # now scale X
    # the further out X is the more we want it to move
    #discrete = m.entities[0].discrete(m.vertices)
    #X[X>0] *= radius / discrete[:,0].max()
    #m.vertices[:,0] = X

    m._cache.clear()

    assert np.allclose(m.vertices[0], 0.0)
    assert np.allclose(m.vertices[-1], [0.0, length])

    plt.plot(*Point([0,length]).buffer(radius).exterior.xy,
             linestyle='dashed')

    plt.scatter(*m.vertices.T)

    m.show()

    """
