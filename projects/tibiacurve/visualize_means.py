#!/usr/bin/python3

import os

from projects.tibiacurve import common

MEANS_OUTPUT_BY_SLM_DIR = os.path.expanduser('~/Dropbox/TIBIA/CURVATURE/results/means/sm%02d')
DATA_BY_SLM_DIR = os.path.expanduser('~/Dropbox/TIBIA/CURVATURE/results/data/sm%02d')
MEANS_OUTPUT_LOG = 'output.txt'


def get_vis_opts(output_dir, radius, diffs):
    if diffs:
        filename = 'diff_' + ('all' if diffs[0][0] == 7 else common.SUBDIRS[diffs[0][0]]) + '_' + common.SUBDIRS[diffs[0][1]]
        filename = os.path.join(output_dir, filename)
    else:
        filename = os.path.join(output_dir, 'all_specimen')
    return dict(
        camera=common.get_camera_vertical(), res=(512 + 320, 4 * 1024),
        colors=common.GROUP_COLORS_MAP,
        default_color=(1, 0, 0),
        radius=radius,
        diffs=diffs,
        normalize=True,
        balls=True,
        arrows_scaling=20,
        filename=[filename + '_frontal.png', filename + '_medial.png']
        )


def generate_means_visualization(input_dir, output_dir, log_file):
    curves_processor = common.get_processor(input_dir, log_file)
    common.mkdir_if_not_exist(output_dir)
    # all
    curves_processor.visualize_means(input_dir, opts=get_vis_opts(output_dir, 0.03, None))

    # all mean diffs
    for i in range(7):
        curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[7, i]]))

    # subsequent means diffs
    for i in range(6):
        curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[i, i + 1]]))

    # 0-3, 3-7, 0-7
    curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[0, 3]]))
    curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[3, 6]]))
    curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[0, 6]]))


def compute_means(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_variability(output_dir)


# different scale for different
for slm in [10, 20, 30]:
    compute_means(slm, DATA_BY_SLM_DIR % slm, MEANS_OUTPUT_LOG)
    generate_means_visualization(DATA_BY_SLM_DIR % slm, MEANS_OUTPUT_BY_SLM_DIR % slm, MEANS_OUTPUT_LOG)
