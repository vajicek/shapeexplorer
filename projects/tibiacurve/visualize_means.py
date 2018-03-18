#!/usr/bin/python3

import os

from projects.tibiacurve import common  

MEANS_OUTPUT_BY_SLM_DIR = '/home/vajicek/Dropbox/TIBIA/CURVATURE/results/means/sm%02d'
MEANS_OUTPUT_LOG = 'output.txt'


def get_vis_opts(output_dir, radius, diffs):
    if diffs:
        filename = 'diff_all_' + common.SUBDIRS[diffs[0][1]]
        filename = os.path.join(output_dir, filename)
    else:
        filename = os.path.join(output_dir, 'all_specimen')
    return dict(
        camera=common.get_camera(),
        res=(4 * 1024, 512),
        colors={"A_eneolit": (1, 0, 0),
               "B_bronz": (0, 1, 0),
               "C_latén": (0, 0, 1),
               "D_raný středověk": (1, 1, 0),
               "E_vrcholný středověk": (1, 0, 1),
               "F_pachner": (0, 1, 1),
               "G_angio": (0, 0, 0),
               "all": (1, 1, 1)
               },
        default_color=(1, 0, 0),
        radius=radius,
        diffs=diffs,
        curve=False,
        filename = [filename + '_frontal', filename + '_medial']
        )

    
def generate_means_visualization(output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    # all
    curves_processor.visualize_means(output_dir, opts=get_vis_opts(output_dir, 0.002, None))
                                               
    # all mean diffs
    for i in range(7):
        curves_processor.visualize_mean_difference(output_dir, opts=get_vis_opts(output_dir, [0.003, 0.010], [[7, i]]))
    

def compute_and_generate_means_visualization(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir)


for slm in [10, 20, 30]:
    compute_and_generate_means_visualization(slm, MEANS_OUTPUT_BY_SLM_DIR % slm, MEANS_OUTPUT_LOG)
    generate_means_visualization(MEANS_OUTPUT_BY_SLM_DIR % slm, MEANS_OUTPUT_LOG)
