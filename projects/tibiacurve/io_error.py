#!/usr/bin/python3

import logging
import sys
from base import processcurves 

#logging.basicConfig(level=logging.DEBUG)

def analyze_io_error_slm(slm):
    sys.stdout = open('/home/vajicek/Dropbox/TIBIA/CURVATURE/results/io_error/output_sm%02d.txt' % slm, 'w')
    processcurves.process_curves(slm, True)
    #processcurves.analyze_io_error('output')
    processcurves.analyze_io_error('/home/vajicek/Dropbox/TIBIA/CURVATURE/results/io_error')

analyze_io_error_slm(10)
analyze_io_error_slm(20)
analyze_io_error_slm(30)

#processcurves.process_curves(30, True)
#processcurves.analyze_io_error("output")
