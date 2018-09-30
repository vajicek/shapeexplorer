#!/usr/bin/python3

import os
import sys

from base import processcurves
from projects.tibiacurve import common

SEXDIMORPHISM_OUTPUT_DIR_BY_SLM = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'sexdimorphism/sm%02d')
DATAFOLDER = os.path.join(common.TARGET_ROOT, r'TIBIA/CURVATURE/Tibie\ CURVES')
GROUP_DIR = ['A_eneolit', 'B_bronz', 'C_latén', 'D_raný středověk',
             'E_vrcholný středověk', 'F_pachner', 'G_angio']
SUBDIRS_PAIRS = [['eneolit males', 'eneolit females'],
                 ['bronz males', 'bronz females'],
                 ['latén males', 'latén females'],
                 ['raný střed males', 'raný střed females'],
                 ['vrchol střed males', 'vrchol střed females'],
                 ['pachner males', 'pachner females'],
                 ['angio males', 'angio females']]

def get_processor(input_dir, output_dir, log_file, group_index):
    common.mkdir_if_not_exist(output_dir)
    if log_file:
        sys.stdout = open(os.path.join(output_dir, log_file), 'w')
    return processcurves.CurvesProcessor(input_dir, SUBDIRS_PAIRS[group_index], None, output_dir)

def analyze_sexual_dimorphism(slm, output_dir, log_file, slm_handling):
    for group_index in range(len(GROUP_DIR)):
        group_output_dir = os.path.join(output_dir, GROUP_DIR[group_index])
        curves_processor = get_processor(DATAFOLDER, group_output_dir, log_file, group_index)
        curves_processor.preprocess_curves(slm, True)
        curves_processor.analyze_variability(group_output_dir, group_output_dir, slm_handling=slm_handling)

for slm in common.SLM_COUNTS:
    for slm_handling in common.SLM_HANDLING:
        output_slm_dir = (SEXDIMORPHISM_OUTPUT_DIR_BY_SLM % slm) + '_' + slm_handling
        analyze_sexual_dimorphism(slm, output_slm_dir, common.OUTPUT_LOG, slm_handling)
