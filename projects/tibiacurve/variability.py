#!/usr/bin/python3

import logging
logging.basicConfig(level=logging.DEBUG)

#import sys
#sys.stdout = open('/home/vajicek/Dropbox/TIBIA/CURVATURE/results/variability/output.txt', 'w')

# process input and io error
from base import processcurves 
processcurves.process_curves()
processcurves.analyze_variability()

