#!/usr/bin/python3

from projects.tibiacurve import common

VARIABILITY_OUTPUT_DIR_BY_SLM = '/home/vajicek/DB/TIBIA/CURVATURE/results/variability/sm%02d'
VARIABILITY_OUTPUT_LOG = 'output.txt'


def analyze_variability_slm(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, None)
    #curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir)


for slm in [20]:
    analyze_variability_slm(slm, VARIABILITY_OUTPUT_DIR_BY_SLM % slm, VARIABILITY_OUTPUT_LOG)
