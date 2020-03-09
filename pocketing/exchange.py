"""
exchange.py
-------------

Format toolpaths as JSON or G-Code (eventually)
"""


def to_json(paths, radius, **kwargs):
    """
    """
    return json.dumps(
        {'radius': radius,
         'path': toolpath.tolist(),
         'bounds': np.reshape(polygon.bounds, (2, 2)).tolist()}, **kwargs)


def from_json(obj):
    """
    """
    pass
