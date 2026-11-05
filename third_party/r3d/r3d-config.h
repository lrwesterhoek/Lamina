/* Vendored stand-in for r3d's CMake-generated header (config/r3d-config.h.in).
   R3D_MAX_VERTS bounds the vertex buffer of one clipped polyhedron. A tetrahedron
   clipped by the 6 planes of one grid cell has at most 10 faces, so at most 16
   vertices (r3d's graph is 3-regular: V = 2F - 4), and a cut holds at most ~40 while
   it runs; 128 leaves a wide margin. The upstream default of 512 cost time for
   nothing: r3d_clip clears a flag per buffer slot for every plane, and the exact
   deposits copy polyhedra per cell (src/CGAL_triangulation/exact_sweep.h). */
#ifndef R3D_CONFIG_H_
#define R3D_CONFIG_H_

#define R3D_MAX_VERTS 128

#endif
