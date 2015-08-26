#!/usr/bin/env python2
import sys, os, array, time, gc
import bpy

sys.path.insert(0, '.') 
from readply import readply

args = []
idx = sys.argv.index('--')
if idx != 1:
    args = sys.argv[idx+1:]

# In foreach_getset() in source/blender/python/intern/bpy_rna.c the
# passed object is checked for supporting the buffer protocol.
# We can use this functionality to pass chunks of memory containing
# vertex and face data, without having to build up a Python data
# structure. We use NumPy arrays here to easily pass the data.

fname = '/data/models/uss_enterprise-jjabrams/export.bin.ply'
#fname = '/home/paulm/mnt/elvis/uva/rbc-util/RBC.00003300.ply'
#fname = 'zeroindex.ply'

if len(args) > 0:
    fname = args[0]

t0 = time.time()

num_vertices, num_faces, varray, farray = readply(fname)

# Create a mesh + object using the binary vertex and face data

mesh = bpy.data.meshes.new(name='imported mesh')

mesh.vertices.add(num_vertices)
mesh.vertices.foreach_set('co', varray)

mesh.tessfaces.add(num_faces)
mesh.tessfaces.foreach_set('vertices_raw', farray)

mesh.validate()
mesh.update()

obj = bpy.data.objects.new('imported object', mesh)

s = bpy.context.scene
s.objects.link(obj)
s.objects.active = obj
obj.select = True

t1 = time.time()
print('Imported in %.3fs' % (t1-t0))

del varray
del farray
gc.collect()