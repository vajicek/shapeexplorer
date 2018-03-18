#!/usr/bin/python3

import os

from projects.tibiacurve import common  

LOADINGS_OUTPUT_BY_SLM_DIR = '/home/vajicek/Dropbox/TIBIA/CURVATURE/results/variability/sm%02d'
LOADINGS_OUTPUT_LOG = 'output.txt'


def get_vis_opts(output_dir, radius, pca_no):
    filename = os.path.join(output_dir, 'pca_' + str(pca_no + 1))
    return dict(
        camera=common.get_camera(),
        res=(4 * 1024, 512),
        colors={ "all": (1, 1, 1) },
        radius=radius,
        pca_no=pca_no,
        filename=[filename + '_frontal', filename + '_medial']
        )

    
def generate_loading_visualization(output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    for i in range(6):
        curves_processor.visualize_loadings(output_dir, opts=get_vis_opts(output_dir, [0.002, 0.008], i))
    

def compute_and_generate_means_visualization(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir)


for slm in [10, 20, 30]:
    compute_and_generate_means_visualization(slm, LOADINGS_OUTPUT_BY_SLM_DIR % slm, LOADINGS_OUTPUT_LOG)
    generate_loading_visualization(LOADINGS_OUTPUT_BY_SLM_DIR % slm, LOADINGS_OUTPUT_LOG)
