readply - a Python extension module for fast(er) import of PLY files in Blender
===============================================================================

Author
------

Paul Melis <paul.melis@surfsara.nl>
SURFsara Visualization group

The files under the rply/ directory are a copy of the RPly 1.1.4 
source distribution (see http://w3.impa.br/~diego/software/rply/)

License
---------

All files, except the ones in the rply/ directory:

XXX

See rply/LICENSE for the the license of the RPly sources.

Rationale
---------

The default PLY importer in Blender is slow (or any Python-based import 
script in Blender for that matter). This is because during import 
Python data structures are built up holding all geometry, vertex colors, 
etc. This simply takes quite a lot of time (and memory).

Fortunately, in foreach_getset() in source/blender/python/intern/bpy_rna.c 
the passed object may support the buffer protocol.
We can use this functionality to pass chunks of memory containing
vertex and face data, without having to build up Python data
structures. We use NumPy arrays in the readply extension module 
to easily pass the data directly to Blender. 

Note: the readply module is not tied to Blender in any way and can 
be used as a general PLY reader for Python.

Performance
-----------

Below are some numbers when importing the Asian Dragon model ([1]) from 
The Stanford 3D Scanning Repository ([2]). This 3D model consists of
3,609,600 vertices and 7,219,045 triangles.

With Blender 2.76 and xyzrgb_dragon.ply already in the filesystem cache:

# Native blender PLY importer (bpy.ops.import_mesh.ply())
$ blender -P test/blender_native_import.py -- xyzrgb_dragon.ply
total                           81.474s

# mesh_readply.py using readply extension module
$ blender -P mesh_readply.py -- xyzrgb_dragon.ply
reaply():                        0.783s
blender mesh+object creation:   12.598s
total                           13.381s

I.e. in this test the mesh_readply.py script (which uses the readply 
module) loads the Dragon model 6.09x faster into Blender than 
Blender's own PLY import script.

Memory usage also improved substantially, as measured by looking at
the peak RSS and VSIZE numbers during the loading:

Native blender PLY importer: 4.164 GB (RSS) | 5.036 GB (VSIZE)
Using readply module:        1.833 GB (RSS) | 3.227 GB (VSIZE)


[1] http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz
[2] http://graphics.stanford.edu/data/3Dscanrep/

Notes
-----

- Make sure that the version of numpy used for compiling the readply
  extension has the same API version as the one used in Blender.
- The extension module can be compiled for both Python 2.x and 3.x,
  even though Blender always uses Python 3.x (at least, modern versions
  of Blender do ;-)).
- Texture coordinates may be stored in s+t or u+v vertex fields 

Bugs
----

- Polygons with more than 4 vertices are not currently supported and 
  will screw up the resulting 3D model
- It is assumed that the PLY file has vertex coordinates defined
  in x, y and z order (the PLY header allows properties in any order).
  