#!/usr/bin/env python2
import sys
sys.path.insert(0, '.')
import readply

p = readply.readply('test/textured_monkey.ply', blender_face_indices=False)
