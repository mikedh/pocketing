# pypocketing: ALPHA

[![Test And Publish](https://github.com/mikedh/pocketing/workflows/Test%20And%20Publish/badge.svg)](https://github.com/mikedh/pocketing/actions?query=workflow%3A%22Test+And+Publish%22) [![PyPI version](https://badge.fury.io/py/pocketing.svg)](https://pypi.org/project/pocketing/)
----------------------





![ScreenShot](https://raw.github.com/mikedh/pypocketing/master/docs/contour_troch.png)

Fill 2D regions with traversals, useful someday for generating milling tool paths.

## Disclaimer: Crusty Alpha Software
This is a dump of a bunch of prototype code. It is only put up in the hopes that it becomes less terrible someday, but until then you should probably use something else: [pyactp](https://github.com/mikedh/pyactp), [openvoronoi](https://github.com/aewallin/openvoronoi), [opencamlib](https://github.com/aewallin/opencamlib), [libarea](https://github.com/Heeks/libarea)


## Design Goals: Why Bother

There are a lot of other options above. However, most of them aren't super active and are generally C- based with python bindings. This is intended as a vectorized numpy approach to the same problem, in the vein of [trimesh](https://github.com/mikedh/trimesh).

## Scope
- Accept shapely.geometry.Polygon objects as input
- Generate toolpath output as a sequence of (n, 2) float arrays
- Collision check and calculate feed rates using raster checks