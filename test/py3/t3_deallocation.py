#!/usr/bin/env python3
import gc, time
from readply import readply

#res = readply('/data/models/uss_enterprise-jjabrams/export.bin.ply')
res = readply('/home/paulm/mnt/elvis/uva/rbc-util/RBC.00003300.ply')

print('Model read')

del res
gc.collect()

print('Collected')

time.sleep(5)