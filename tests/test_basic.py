import os
import trimesh
import unittest
import pocketing

import numpy as np


def get_model(file_name):
    """
    Load a model from the models directory by expanding paths out.

    Parameters
    ------------
    file_name : str
      Name of file in `models`

    Returns
    ------------
    mesh : trimesh.Geometry
      Trimesh object or similar
    """
    pwd = os.path.dirname(os.path.abspath(
        os.path.expanduser(__file__)))
    return trimesh.load(os.path.abspath(
        os.path.join(pwd, '../models', file_name)))


class PocketTest(unittest.TestCase):

    def test_contour(self):
        path = get_model('wrench.dxf')
        poly = path.polygons_full[0]
        # generate tool paths
        toolpaths = pocketing.contour.contour_parallel(poly, .05)

        assert all(trimesh.util.is_shape(i, (-1, 2))
                   for i in toolpaths)

    def test_troch(self):
        path = get_model('wrench.dxf')
        polygon = path.polygons_full[0]

        # set radius arbitrarily
        radius = .125
        # set step to 10% of tool radius
        step = radius * 0.10
        # generate our trochoids
        toolpath = pocketing.trochoidal.toolpath(
            polygon, step=step)

        assert trimesh.util.is_shape(toolpath, (-1, 2))

    def test_archimedian(self):
        # test generating a simple archimedean spiral
        spiral = pocketing.spiral.archimedean(0.5, 2.0, 0.125)
        assert trimesh.util.is_shape(spiral, (-1, 3, 2))

    def test_helix(self):
        # check a 3D helix

        # set values off a tool radius
        tool_radius = 0.25
        radius = tool_radius * 1.2
        pitch = tool_radius * 0.3
        height = 2.0

        # create the helix
        h = pocketing.spiral.helix(
            radius=radius,
            height=height,
            pitch=pitch,)

        # should be 3-point arcs
        check_arcs(h)

        # heights should start and end correctly
        assert np.isclose(h[0][0][2], 0.0)
        assert np.isclose(h[-1][-1][2], height)

        # check the flattened 2D radius
        radii = np.linalg.norm(h.reshape((-1, 3))[:, :2], axis=1)
        assert np.allclose(radii, radius)


def check_arcs(arcs):
    # arcs should be 2D or 2D 3-point arcs
    assert trimesh.util.is_shape(arcs, (-1, 3, (3, 2)))
    # make sure arcs start where previous arc begins
    for a, b in zip(arcs[:-1], arcs[1:]):
        assert np.allclose(a[2], b[0])


if __name__ == '__main__':
    unittest.main()
