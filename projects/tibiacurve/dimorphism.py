#!/usr/bin/python3

import os
import sys

from base import processcurves
from projects.tibiacurve import common

SEXDIMORPHISM_OUTPUT_DIR_BY_SLM = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'sexdimorphism/sm%02d_%s')
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

def get_processor(input_dir, output_dir, log_file, group_index, verbose=False):
    common.mkdir_if_not_exist(output_dir)
    if log_file and not verbose:
        sys.stdout = open(os.path.join(output_dir, log_file), 'w')
    return processcurves.CurvesProcessor(input_dir, SUBDIRS_PAIRS[group_index], None, output_dir)

def analyze_sexual_dimorphism(output_dir, log_file, slm_handling, slm, args):
    for group_index in range(len(GROUP_DIR)):
        group_output_dir = os.path.join(output_dir, GROUP_DIR[group_index])
        curves_processor = get_processor(DATAFOLDER, group_output_dir, log_file, group_index, args.verbose)
        curves_processor.preprocess_curves(slm, True)
        curves_processor.analyze_variability(group_output_dir, group_output_dir, slm_handling=slm_handling)

common.process_lms_handling(SEXDIMORPHISM_OUTPUT_DIR_BY_SLM, analyze_sexual_dimorphism, common.args())
