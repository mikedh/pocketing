import os
import trimesh
import unittest

import pocketing


def get_model(file_name):
    """
    Load a model from the models directory by expanding paths out.
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
        # test generating a simple archimedian spiral
        spiral = pocketing.spiral.archimedian(0.5, 2.0, 0.125)
        assert trimesh.util.is_shape(spiral, (-1, 2))


if __name__ == '__main__':
    unittest.main()
