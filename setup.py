from distutils.core import setup, Extension

module1 = Extension('readply', 
    include_dirs = ['./rply'],
    sources = ['readply.cpp', 'rply/rply.c'])

setup(
    name = 'readply',
    author = 'Paul Melis',    
    author_email = 'paul.melis@surfsara.nl',
    version = '1.0',
    description = 'Load PLY files into NumPy arrays',
    ext_modules = [module1]
)

