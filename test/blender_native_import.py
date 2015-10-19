# To be run from within Blender
# $ blender -P blender_native_import.py -- <file.ply>
import sys, time
import bpy

args = []
idx = sys.argv.index('--')
if idx != 1:
    args = sys.argv[idx+1:]

t0 = time.time()

bpy.ops.import_mesh.ply(filepath=args[0])

t1 = time.time()
print('Imported in %.3fs' % (t1-t0))