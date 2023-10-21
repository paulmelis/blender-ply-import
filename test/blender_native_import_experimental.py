# To be run from within Blender
# $ blender -P blender_native_import.py -- <file.ply>
import sys, time
import bpy

args = []
idx = sys.argv.index('--')
if idx != 1:
    args = sys.argv[idx+1:]

t0 = time.time()

# New experimental, but much faster
bpy.ops.wm.ply_import(filepath=args[0])

t1 = time.time()
print('Imported in %.3fs (experimental)' % (t1-t0))
