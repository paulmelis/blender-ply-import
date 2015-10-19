from distutils.core import setup, Extension
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
