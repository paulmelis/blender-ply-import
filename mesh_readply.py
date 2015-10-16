#!/usr/bin/env python2
#
# blender -P mesh_readply.py -- file.ply
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

num_vertices, num_faces, varray, farray, vnarray, vcolarray = readply(fname)

t1 = time.time()
print('PLY file read in %.3fs' % (t1-t0))

# Create a mesh + object using the binary vertex and face data

mesh = bpy.data.meshes.new(name='imported mesh')

mesh.vertices.add(num_vertices)
mesh.vertices.foreach_set('co', varray)

mesh.tessfaces.add(num_faces)
mesh.tessfaces.foreach_set('vertices_raw', farray)

mesh.validate()
mesh.update()

if vcolarray is not None:
    
    """    
    # For each face, set the vertex colors of the vertices making up that face
    for fi in range(num_faces):
        
        # Get vertex indices for this triangle/quad
        i, j, k, l = farray[4*fi:4*fi+4]
        
        face_col = vcol_data[fi]
        face_col.color1 = vcolarray[3*i:3*i+3]
        face_col.color2 = vcolarray[3*j:3*j+3]
        face_col.color3 = vcolarray[3*k:3*k+3]
        if l != 0:
            face_col.color4 = vcolarray[3*l:3*l+3]
    """
    
    vcol_layer = mesh.vertex_colors.new()
    vcol_data = vcol_layer.data
    vcol_data.foreach_set('color', vcolarray)

if vnarray is not None:
    print('Warning: NOT applying vertex normals (yet)')
    
mesh.validate()
mesh.update()

obj = bpy.data.objects.new('imported object', mesh)

s = bpy.context.scene
s.objects.link(obj)
s.objects.active = obj
obj.select = True

t2 = time.time()
print('Blender object+mesh created in %.3fs' % (t2-t1))

del varray
del farray

print('Total import time %.3fs' % (t2-t0))
