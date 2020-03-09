"""
graph.py
---------------

General graph functions useful in toolpath generation.
"""

import trimesh
import collections

import numpy as np
import networkx as nx

from scipy import spatial


def traverse_child_first(g, closest=None):
    """
    Traverse a tree, with the rule that a node can only be visted
    once all of its children are visited.

    Useful for linking toolpaths.

    Arguments
    -----------
    g : networkx.DiGraph
        Source graph
    closest: function
        Optional for determining closest leaves

    Assumptions
    -------------
    - Graph is a tree


    Example Graph
    --------------
      A
     B   C
    D E    F
            G
             H
            I  J

    g = nx.DiGraph()
    g.add_edges_from((('A','B'),
                      ('B','D'),
                      ('B','E'),
                      ('A','C'),
                      ('C','F'),
                      ('F','G'),
                      ('G','H'),
                      ('H','I'),
                      ('H','J')))


    Correct Result (one of several):
    D E B I J H G F C A

    Algorithm
    ----------
    -  start at a leaf
    -  get all sibling nodes (children of same parent)
    -- see which siblings haven't been visited
    -- for each unvisited sibling
    ----- if sibling is a leaf, add it to the traversal and
          check the next sibling
    ----- if sibling is not a leaf, find ONE leaf below it
          and move there
    ------- you can pick any leaf below it or use a distance metric
    -- if you have no unvisited siblings, move up to the parent
    -- once you reach the root node, stop

    """

    assert nx.is_forest(g)

    leaves = [k for k in g.nodes.keys() if len(g.edges(k)) == 0]
    root = [n for n in g.nodes() if
            len(list(g.predecessors(n))) == 0][0]

    # for each node remember names of all leaf nodes below it
    for n in g.nodes():
        g.nodes[n]['leaves'] = []
    for l in leaves:
        for n in nx.shortest_path(g, root, l):
            if n == l:
                continue
            g.nodes[n]['leaves'].append(l)

    # start the traversal at an arbitrary node
    traversal = collections.deque([leaves[0]])

    # stop once the traversal reaches the root node
    while traversal[-1] != root:
        current = traversal[-1]
        # this is a tree, so each node has ONE parent
        parent = next(iter(g.predecessors(current)))
        # children of our parent that are not ourselves
        siblings = list(set(g.successors(parent)).difference(
            {current}))

        # which of our siblings have we not visited yet
        unvisited = [s for s in siblings if s not in traversal]

        if len(unvisited) == 0:
            # if we have no unvisited siblings we can move up the tree
            traversal.append(parent)
        else:
            for u in unvisited:
                if u in leaves:
                    # if our unvisited sibling is a leaf, we add it
                    # and then keep going until we consume all elements
                    # or find an unvisited sibling that is not a leaf
                    traversal.append(u)
                else:
                    # a list of all leaves associated with the unvisited node
                    c_leaf = g.nodes[u]['leaves']
                    # only append one of them now, and break
                    if closest is None:
                        next_node = c_leaf[0]
                    else:
                        next_node = closest(g,
                                            traversal[-1],
                                            c_leaf)
                    traversal.append(next_node)
                    break
    return list(traversal)


def graph_to_paths(g, traversal, polygons):
    """
    Turn a graph and traversal into a connected path.

    Parameters

    """
    leaves = [k for k in g.nodes.keys() if len(g.edges(k)) == 0]

    paths = collections.deque()

    for t in traversal:
        current = np.array(polygons[t].exterior.coords)
        ccw = (trimesh.path.util.is_ccw(current) * 2) - 1
        current = trimesh.path.polygons.resample_path(
            current,
            step=.02)[::ccw]

        if t in leaves:
            # we are starting a new sequence of paths
            paths.append(collections.deque([current]))
        else:
            # we are finding the

            previous = paths[-1][-1]

            tree = spatial.cKDTree(current)
            distance, index = tree.query(previous[-1], k=1)

            rolled = np.roll(current[:-1], -index, axis=0)
            rolled = np.vstack((rolled, [rolled[0]]))

            paths[-1].append(rolled)

    paths = [np.vstack(i) for i in paths]
    return paths


def is_1D_graph(g):
    """
    Check a graph to see if it is 1D, or
    a chain of nodes with one or zero parents
    and children.

    Parameters
    ------------
    g : networkx Graph

    Returns
    ------------
    is_1D : bool
        Is graph 1D or not
    """
    # check degree of sucessors
    for v in g.succ.values():
        if len(v) not in [0, 1]:
            return False

    # check degree of predecessors
    for v in g.pred.values():
        if len(v) not in [0, 1]:
            return False

    # made it through all checks
    return True
