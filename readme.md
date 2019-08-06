# readply - a Python extension module for fast(er) import of PLY files in Blender

## Author

Paul Melis <paul.melis@surfsara.nl>

SURFsara Visualization group

The files under the `rply/` directory are a copy of the RPly 1.1.4 
source distribution (see http://w3.impa.br/~diego/software/rply/).

## License

See the `LICENSE` file in the root directory of this distribution,
which applies to all files except the ones in the `rply/` directory.

See `rply/LICENSE` for the the license of the RPly sources.

## Rationale

Note: this module has NOT been updated for Blender 2.80 yet. 
It only works with Blender 2.7x at the moment.

The default PLY importer in Blender is slow (or any Python-based import 
script in Blender for that matter). This is because during import 
Python data structures are built up holding all geometry, vertex colors, 
etc. This simply takes quite a lot of time (and memory).

Fortunately, in `foreach_getset()` in `source/blender/python/intern/bpy_rna.c` 
the passed object may support the buffer protocol.
We can use this functionality to pass chunks of memory containing
vertex and face data, without having to build up Python data
structures. We use NumPy arrays in the readply extension module 
to easily pass the data directly to Blender. 

The readply module is not tied to Blender in any way and can 
be used as a general PLY reader in Python.

## Performance

Below are some numbers when importing the Asian Dragon model ([1]) from 
The Stanford 3D Scanning Repository ([2]). This 3D model consists of
3,609,600 vertices and 7,219,045 triangles.

With Blender 2.76 and xyzrgb_dragon.ply already in the filesystem cache:

```
# Native blender PLY importer (bpy.ops.import_mesh.ply())
$ blender -P test/blender_native_import.py -- xyzrgb_dragon.ply
total                           81.474s

# mesh_readply.py using readply extension module
$ blender -P mesh_readply.py -- xyzrgb_dragon.ply
reaply():                        0.783s
blender mesh+object creation:   12.598s
total                           13.381s
```

I.e. in this test the mesh_readply.py script (which uses the readply 
module) loads the Dragon model 6.09x faster into Blender than 
Blender's own PLY import script.

Memory usage also improved substantially, as measured by looking at
the peak RSS and VSIZE numbers during the loading:

```
Native blender PLY importer: 4.164 GB (RSS) | 5.036 GB (VSIZE)
Using readply module:        1.833 GB (RSS) | 3.227 GB (VSIZE)
```

1. http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz
2. http://graphics.stanford.edu/data/3Dscanrep/

## Building

Two small shell scripts are provided to easily build an in-place version
of the Python module:

- `make2.sh` for Python 2.x
- `make3.sh` for Python 3.x

These simply call `setup.py` with a few options to do the in-place building.

For use with Blender use the `setup.py` script as

```
$ blender -b -P setup.py
```

This should use Blender's specific version of Python to build the module.

## Notes

- Make sure that the version of NumPy used for compiling the readply
  extension has the same API version as the one that is used by Blender
  (the official binary distributions of Blender include a version of NumPy)
- The `readply` extension module can be compiled for both Python 2.x and 3.x,
  even though Blender uses Python 3.x
- Texture coordinates may be stored in s+t or u+v vertex fields, depending
  on what property names the PLY file being read uses

## Bugs

- Polygons with more than 4 vertices are not currently supported and 
  will screw up the resulting 3D model
- It is assumed that if the PLY file includes vertex coordinates they 
  are defined in x, y and z order (the PLY header allows properties in any order).
  
  
Readply API
-----------

numpy arrays returned are all 1-dimensional. i.e. vertex coordinates
are NOT returned as an Nx3 array but as length 3*N.

farray: 4 indices per triangle/quad. triangles will have last index set to 0
