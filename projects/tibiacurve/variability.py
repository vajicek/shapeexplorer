#!/usr/bin/python3

import os

from projects.tibiacurve import common

VARIABILITY_OUTPUT_DIR_BY_SLM = os.path.join(common.TARGET_ROOT, 'TIBIA/CURVATURE/results/variability/sm%02d')
VARIABILITY_OUTPUT_LOG = 'output.txt'


def analyze_variability_slm(slm, output_dir, log_file, slm_handling):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir, output_dir, slm_handling=slm_handling)


for slm in [10, 20, 30]:
#for slm in [20]:
    for slm_handling in ["none", "procd", "bende"]:
    #for slm_handling in ["none"]:
        output_slm_dir = (VARIABILITY_OUTPUT_DIR_BY_SLM % slm) + "_" + slm_handling
        analyze_variability_slm(slm, output_slm_dir, VARIABILITY_OUTPUT_LOG, slm_handling)
