#!/usr/bin/env python3
import sys, time
from readply import readply

t0 = time.time()

res = readply(sys.argv[1])

t1 = time.time()
print('Read in %.1fs' % (t1-t0))
