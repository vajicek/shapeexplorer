#!/usr/bin/python3

import os

from projects.tibiacurve import common
from base import rscriptsupport

OUTPUT_DIR = os.path.expanduser('~/Dropbox/TIBIA/CURVATURE/results/length_analysis')
OUTPUT_LOG = 'output.txt'


curves_processor = common.get_processor(OUTPUT_DIR, OUTPUT_LOG)
curves, names = curves_processor.load_all_curves(None)

curves_lengths = curves_processor.measure_length(curves)
rscript = rscriptsupport.RScripInterface(OUTPUT_DIR)
rscript.write_csv('curves_lengths.csv', curves_lengths)

curves_processor.length_analysis(OUTPUT_DIR)
