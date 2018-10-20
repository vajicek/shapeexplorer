#!/usr/bin/python3

import os
import logging

from projects.tibiacurve import common

MEANS_OUTPUT_BY_SLM_DIR = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'means/sm%02d_%s')
DATA_BY_SLM_DIR = os.path.join(common.TARGET_ROOT, common.RESULT_FOLDER, 'data/sm%02d_%s')

def get_vis_opts(output_dir, radius, diffs, count=None):
    if diffs:
        filename = 'diff_' + ('all' if diffs[0][0] == count else common.SUBDIRS[diffs[0][0]]) + '_' + common.SUBDIRS[diffs[0][1]]
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

def generate_means_visualization(input_dir, output_dir, log_file, args):
    curves_processor = common.get_processor(input_dir, log_file)
    common.mkdir_if_not_exist(output_dir)

    logging.info('Visualize all means')
    curves_processor.visualize_means(input_dir, opts=get_vis_opts(output_dir, 0.03, None))

    groups_count = curves_processor.get_groups_count()
    females_shift = int(groups_count / 2)

    logging.info('Visualize group mean diffs to all mean')
    for i in range(groups_count):
        curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[groups_count, i]], groups_count))

    logging.info('Visualize subsequent means diffs')
    for sex_name, sex_shift in dict(male=0, female=females_shift).items():
        logging.info('Visualize subsequent means diffs for %s' % sex_name)
        for i in range(sex_shift, sex_shift + females_shift - 1):
            curves_processor.visualize_mean_difference(input_dir,
                opts=get_vis_opts(output_dir,
                    [0.03, 0.10],
                    [[i, i + 1]],
                    groups_count))

    logging.info('Visualize 0-3, 3-6, 0-6')
    for sex_shift in (0, females_shift):
        for groups in ([0, 3], [3, 6], [0, 6]):
            curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(
                output_dir, [0.03, 0.10], [[sex_shift + groups[0], sex_shift + groups[1]]]))

    logging.info('Visualize sex diffs for 3, 5')
    for group in (3, 5):
        curves_processor.visualize_mean_difference(input_dir, opts=get_vis_opts(output_dir, [0.03, 0.10], [[group, females_shift + group]]))

def compute_means(slm, output_dir, log_file, slm_handling, args):
    curves_processor = common.get_processor(output_dir, log_file, args.verbose)
    logging.info('Preprocess curves')
    curves_processor.preprocess_curves(slm, True)
    logging.info('Analyze variability')
    curves_processor.analyze_variability(output_dir, output_dir, slm_handling=slm_handling)

def process(output_slm_dir, log_file, slm_handling, slm, args):
    means_output_dir = MEANS_OUTPUT_BY_SLM_DIR % (slm, slm_handling)
    compute_means(slm, output_slm_dir, log_file, slm_handling, args)
    generate_means_visualization(output_slm_dir, means_output_dir, log_file, args)

common.process_lms_handling(DATA_BY_SLM_DIR, process, common.args())
