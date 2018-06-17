#!/usr/bin/python3

import os

from projects.tibiacurve import common

VARIABILITY_OUTPUT_DIR_BY_SLM = os.path.expanduser('~/Dropbox/TIBIA/CURVATURE/results/variability/sm%02d')
VARIABILITY_OUTPUT_LOG = 'output.txt'


def analyze_variability_slm(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir)


for slm in [10, 20, 30]:
    analyze_variability_slm(slm, VARIABILITY_OUTPUT_DIR_BY_SLM % slm, VARIABILITY_OUTPUT_LOG)
