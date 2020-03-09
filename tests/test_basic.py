import os
import trimesh
import unittest


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
        import pocketing

        path = get_model('wrench.dxf')
        poly = path.polygons_full[0]
        # generate tool paths
        toolpaths = pocketing.contour.contour_parallel(poly, .05)

        assert all(trimesh.util.is_shape(i, (-1, 2))
                   for i in toolpaths)

    def test_contour(self):
        import pocketing

        path = get_model('wrench.dxf')
        polygon = path.polygons_full[0]

        radius = .125 * 25.4
        step = radius * 0.10

        toolpath = pocketing.trochoidal.toolpath(
            polygon, step=step)

        assert trimesh.util.is_shape(toolpath, (-1, 2))


if __name__ == '__main__':
    unittest.main()
