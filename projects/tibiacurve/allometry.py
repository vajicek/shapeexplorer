#!/usr/bin/python3

import os

from projects.tibiacurve import common

ALLOMETRY_OUTPUT_DIR_BY_SLM = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'allometry/sm%02d_%s')

def analyze_allometry_slm(output_dir, log_file, slm_handling, slm, args):
    curves_processor = common.get_processor(output_dir, log_file, args.verbose, False)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_allometry(output_dir, output_dir, slm_handling=slm_handling)

common.process_lms_handling(ALLOMETRY_OUTPUT_DIR_BY_SLM, analyze_allometry_slm, common.args())
