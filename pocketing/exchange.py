"""
exchange.py
-------------

Format toolpaths as JSON or G-Code (eventually)
"""

import json
import numpy as np


def to_json(path, radius, polygon, **kwargs):
    """
    """
    return json.dumps(
        {'radius': radius,
         'path': path.tolist(),
         'bounds': np.reshape(
             polygon.bounds, (2, 2)).tolist()}, **kwargs)


def from_json(obj):
    """
    """
    pass
