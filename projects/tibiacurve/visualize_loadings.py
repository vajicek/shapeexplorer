#!/usr/bin/python3

import os

from projects.tibiacurve import common

LOADINGS_OUTPUT_BY_SLM_DIR = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'variability/sm%02d')
DATA_BY_SLM_DIR = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'data/sm%02d')

def get_vis_opts(output_dir, radius, pca_no):
    filename = os.path.join(output_dir, 'pca_' + str(pca_no + 1))
    return dict(
        camera=common.get_camera_vertical(), res=(512 + 256, 4 * 1024),
        colors={ "all": (1, 1, 1) },
        radius=radius,
        pca_no=pca_no,
        normalize=True,
        arrows_scaling=0.2,
        filename=[filename + '_frontal.png', filename + '_medial.png']
        )

def generate_loading_visualization(input_dir, output_dir, log_file):
    curves_processor = common.get_processor(input_dir, log_file)
    common.mkdir_if_not_exist(output_dir)
    for i in range(curves_processor.get_groups_count() - 1):
        curves_processor.visualize_loadings(input_dir, output_dir, opts=get_vis_opts(output_dir, [0.03, 0.1], i))

def compute_variability(slm, output_dir, log_file, slm_handling):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir, output_dir, slm_handling=slm_handling)

for slm in common.SLM_COUNTS:
    for slm_handling in common.SLM_HANDLING:
        output_slm_dir = (DATA_BY_SLM_DIR % slm) + "_" + slm_handling
        loadings_output_dir = (LOADINGS_OUTPUT_BY_SLM_DIR % slm) + "_" + slm_handling
        compute_variability(slm, output_slm_dir, common.OUTPUT_LOG, slm_handling)
        generate_loading_visualization(output_slm_dir, loadings_output_dir, common.OUTPUT_LOG)
