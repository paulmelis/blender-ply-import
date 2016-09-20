#!/usr/bin/env python2
# blender -P mesh_readply.py -- file.ply
import sys, os, array, time, gc
import bpy, bmesh

# For uv coordinate handling, see http://blender.stackexchange.com/questions/4820/exporting-uv-coordinates

sys.path.insert(0, '.')
try:
    from readply import readply
except ImportError:
    scriptdir = os.path.split(os.path.abspath(__file__))[0]
    sys.path.insert(0, scriptdir)
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

p = readply(fname)

t1 = time.time()
print('PLY file read by readply() in %.3fs' % (t1-t0))

# Create a mesh + object using the vertex and face data in the numpy arrays

mesh = bpy.data.meshes.new(name='imported mesh')

mesh.vertices.add(p['num_vertices'])
mesh.vertices.foreach_set('co', p['vertices'])

mesh.tessfaces.add(p['num_faces'])
mesh.tessfaces.foreach_set('vertices_raw', p['faces'])

mesh.validate()
mesh.update()

if 'vertex_colors' in p:   
    vcol_layer = mesh.vertex_colors.new()
    vcol_data = vcol_layer.data
    vcol_data.foreach_set('color', p['vertex_colors'])
    
mesh.validate()
mesh.update()   

if 'vertex_normals' in p:    
    mesh.vertices.foreach_set('normal', p['vertex_normals'])

if 'texcoords' in p:
    
    texcoords = p['texcoords']
    texcoords = texcoords.reshape((texcoords.size//2, 2))
    
    mesh.uv_textures.new('default')
    
    bm = bmesh.new()
    bm.from_mesh(mesh)
    
    uv_layer = bm.loops.layers.uv[0]
    
    for fi, f in enumerate(bm.faces):
        for l in f.loops:
            vi = l.vert.index
            l[uv_layer].uv = tuple(texcoords[vi])
    
    bm.to_mesh(mesh)

mesh.validate()
mesh.update()   

obj = bpy.data.objects.new('imported object', mesh)

s = bpy.context.scene
s.objects.link(obj)
s.objects.active = obj
obj.select = True

t2 = time.time()
print('Blender object+mesh created in %.3fs' % (t2-t1))

del p

print('Total import time %.3fs' % (t2-t0))
