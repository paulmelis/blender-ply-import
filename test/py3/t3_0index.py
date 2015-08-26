#!/usr/bin/env python3
import os, time
from struct import pack
from readply import readply

if not os.path.isfile('zeroindex.ply'):
    f = open('zeroindex.ply', 'wb')
    f.write(b'''ply
format binary_little_endian 1.0
element vertex 4
property float x
property float y
property float z
element face 1
property list uchar uint vertex_indices
end_header\n''')
    vertices = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (1, 0, 0)
    ]
    for v in vertices:
        f.write(pack('<fff', *v))
    faces = [
        (1, 2, 3, 0),
    ]
    for face in faces:
        f.write(pack('<b', len(face)))
        for vi in face:
            f.write(pack('<i', vi))
    f.close()


res = readply('zeroindex.ply')

print (res)