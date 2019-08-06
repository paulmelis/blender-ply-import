try:
    import bpy
    # We're running in Blender, hack sys.argv to make setup() below
    # do the right thing using Blender's version of Python
    import sys
    sys.argv = ['setup.py', 'build_ext', '--inplace']
    
except ImportError:
    # Running outside blender, work as regular setup.py script
    pass

from setuptools import setup, Extension
import numpy

module1 = Extension('readply', 
    include_dirs = ['./rply', numpy.get_include()],
    sources = ['readply.cpp', 'rply/rply.c'])

setup(
    name = 'readply',
    author = 'Paul Melis',    
    author_email = 'paul.melis@surfsara.nl',
    version = '1.0',
    description = 'Load PLY files into NumPy arrays, for faster Blender import',
    ext_modules = [module1]
)
