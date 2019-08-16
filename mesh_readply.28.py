#!/usr/bin/env python
# Blender 2.8x version
#
# blender -P mesh_readply.py -- file.ply
#
import sys, os, time
import bpy, bmesh

# For uv coordinate handling, see http://blender.stackexchange.com/questions/4820/exporting-uv-coordinates

sys.path.insert(0, '.')
try:
    from readply import readply
except ImportError:
    scriptdir = os.path.split(os.path.abspath(__file__))[0]    
    sys.path.insert(0, scriptdir)
    from readply import readply

# Option parsing

args = []
idx = sys.argv.index('--')
if idx != -1:
    args = sys.argv[idx+1:]

fname = './textured_monkey.ply'
if len(args) > 0:
    fname = args[0]

# Start parsing the PLY file

t0 = time.time()

p = readply(fname)

t1 = time.time()
print('PLY file read by readply() in %.3fs' % (t1-t0))

# Create a mesh + object using the vertex and face data in the numpy arrays

mesh = bpy.data.meshes.new(name='imported mesh')

mesh.vertices.add(p['num_vertices'])
mesh.vertices.foreach_set('co', p['vertices'])

mesh.loops.add(len(p['faces']))
mesh.loops.foreach_set('vertex_index', p['faces'])

mesh.polygons.add(p['num_faces'])
mesh.polygons.foreach_set('loop_start', p['loop_start'])
mesh.polygons.foreach_set('loop_total', p['loop_length'])

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

"""
if 'texcoords' in p:
    
    # XXX This way of assigning UVs is potentially pretty slow for 
    # large numbers of vertices
    
    texcoords = p['texture_coordinates']
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
"""

mesh.validate()
mesh.update()   

obj = bpy.data.objects.new('imported object', mesh)

# Add object to the scene
scene = bpy.context.scene
scene.collection.objects.link(obj)

# Select the new object and make it active
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

t2 = time.time()
print('Blender object+mesh created in %.3fs' % (t2-t1))

del p

print('Total import time %.3fs' % (t2-t0))
