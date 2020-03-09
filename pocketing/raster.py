"""
raster.py
---------------

Collision check tool paths and evaluate chip loading using
rasterized tests.
"""
import zlib

import trimesh
import collections
import numpy as np
import networkx as nx

from PIL import Image, ImageDraw

from scipy import spatial
from shapely.geometry import Polygon


def check_loading(polygon, paths, pixel_count=1e6):
    """
    Check the chip loading of 2D tool paths.

    NOT FUNCTIONAL
    """

    resolution_final = np.array([1920, 1080]) * 2
    resolution_multiplier = 4
    resolution = resolution_final * resolution_multiplier

    space_to_pixel = int(np.round((resolution /
                                   d.extents).min() * .8))
    space_offset = (d.bounds[0] * space_to_pixel).astype(int)
    space_offset -= ((resolution - (space_to_pixel *
                                    d.extents)) / 2).astype(int)

    images = collections.deque()

    im = Image.new(mode='RGB',
                   size=tuple(resolution),
                   color=(255, 255, 255))

    draw = ImageDraw.Draw(im)

    for i in paths:
        pix = ((i * space_to_pixel) - space_offset).astype(int)
        draw.line([tuple(i) for i in pix], fill=(
            255, 0, 0), width=resolution_multiplier * 2)

    i = im.resize(resolution_final, resample=Image.LANCZOS)
    i.save('hi.png')
