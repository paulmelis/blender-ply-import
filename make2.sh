#!/bin/sh
export CFLAGS="-O0 -g"
python2 setup.py build_ext --inplace 
