# readply - a Python extension module for fast(er) import of PLY files in Blender

## Introduction

The default PLY importer in Blender is quite slow when it comes to 
import large mesh files. Most Python-based importers suffer from this
because during import Python data structures are built up holding all 
geometry, vertex colors, etc. This simply takes quite a lot of time 
(and memory). 

Fortunately, Python objects that support the buffer protocol can be
passed in certain places of the Blender Python API.
We can use this functionality to pass chunks of memory containing
vertex and face data, without having to build up Python data
structures. We use NumPy arrays in the `readply` extension module 
to easily pass the data directly to Blender. 

Example usage:
```
$ blender [<scene.blend>] -P mesh_readply.py -- myfile.ply
```

Notes: 

- The `readply` module is not tied to Blender in any way and can 
  be used as a general PLY reader in Python.
- Compared to 2.7 the 2.8x version of Blender already improves on the
  import time in the example below. Memory usage in 2.8x also improved 
  substantially. But the speed improvement using this module is still 
  of the same order (roughly 6x). 
- There was a [Google Summer of Code 2019 project](https://devtalk.blender.org/t/gsoc-2019-fast-import-and-export/7343) for creating new
  faster Blender importers for PLY, STL and OBJ. The results unfortunately
  do not seem to be merged with mainline Blender at the moment.
  If that ever happens this module will probably become obsolete as
  far as its use in Blender is concerned.
- Development and testing is done on Linux, but the module should compile
  and work under different operating systems

## Performance

Below are some numbers when importing the Asian Dragon model [1] from 
The Stanford 3D Scanning Repository [2]. This 3D model consists of
3,609,600 vertices and 7,219,045 triangles.

With Blender 2.80.75 and `xyzrgb_dragon.ply` already in the filesystem cache:

```
# Native blender PLY importer (bpy.ops.import_mesh.ply())
$ blender -P test/blender_native_import.py -- xyzrgb_dragon.ply
total                           38.664 sec

# mesh_readply.py using readply extension module
$ blender -P mesh_readply.py -- xyzrgb_dragon.ply
reaply():                        0.613s
blender mesh+object creation:    4.841s
total                            5.454s
```

I.e. in this test the `mesh_readply.py` script (which uses the `readply`
module) loads the Dragon model 6.4x faster into Blender than 
Blender's own PLY import script.

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

- The module is currently not usable as a drop-in replacement of the
  built-in PLY import in Blender
- It is assumed that if the PLY file includes vertex coordinates they 
  are defined in x, y and z order (the PLY header allows properties in any order).
  
## Author

Paul Melis <paul.melis@surfsara.nl>

SURFsara Visualization group

The files under the `rply/` directory are a copy of the RPly 1.1.4 
source distribution (see http://w3.impa.br/~diego/software/rply/).

## License

See the `LICENSE` file in the root directory of this distribution,
which applies to all files except the ones in the `rply/` directory.

See `rply/LICENSE` for the the license of the RPly sources.

