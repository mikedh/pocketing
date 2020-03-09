# Algorithms

Quick discussion of some pocketing algorithms.


## Trochoid Algorithm

A bunch of tiny little circles expanding in wavefronts along
the medial axis of the polygon.

1. Compute the medial axis of the polygon  
2. Generate a traversal of the medial axis starting with depth first search
  1. Note when the path jumps between non-connected nodes
  2. For parts of the traversal along edges, add them to a 'slow' group
  3. For jumps in the traversal compute a shortest path connecting the jump, then
    *  if the shortest path is below a threshold add this path to the "slow" group
    * if the shortest path is above the threshold add it to the "fast" group
3. For each section of the traversal with no jumps ("slow" group) compute the trochoid-like-curve:
    a. Discretize the path section with linearly increasing
       path distances
    b. compute the largest radius a circle at each point can be
       and still be contained by the polygon
    c. find the subset of circles which have advancing fronts:
       circles centered at p1/p2 with radius r1/r2
       step = |p2-p1| - r1 + r2
    d. find the distance along the section of the path for each of
       the subset of circles
    e. Fit a smoothed function (aka cubic spline) to the distances
       along the path
    f. Sample the smoothed function for the new distances along
       the path
    g. Take the distances, and convert them to XY positions
    h. Find the new radius at each of the XY positions
        * Set theta equal to a linearly increasing value between
           0.0 and `(len(circle subset) * pi * 2.0)`
        * Compute a trochoid with the XY positions, radii, and theta

4. For each section of the traversal which represents a jump ('fast' group) connect to the trochoid on both ends
5. Calculate the 'chip loading', or material area removed per unit distance
6. For regions of the traversal with no chip loading, shortcut the path
7. Generate velocities as a function of chip loading
