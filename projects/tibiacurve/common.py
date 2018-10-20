
import argparse
import logging
import os
import sys

from base import processcurves

#TARGET_ROOT = os.path.expanduser('~/Dropbox')
TARGET_ROOT = os.path.expanduser('~/DB')
RESULT_FOLDER = 'TIBIA/CURVATURE/results_separated'
DATAFOLDER = os.path.join(TARGET_ROOT, r'TIBIA/CURVATURE/Tibie\ CURVES')
OUTPUT_LOG = 'output.txt'

logging.basicConfig(level=logging.INFO)


#SUBDIRS = ['A_eneolit', 'B_bronz', 'C_latén', 'D_raný středověk',
#           'E_vrcholný středověk', 'F_pachner', 'G_angio']
SUBDIRS = [
    'eneolit males',
    'bronz males',
    'latén males',
    'raný střed males',
    'vrchol střed males',
    'pachner males',
    'angio males',
    'eneolit females',
    'bronz females',
    'latén females',
    'raný střed females',
    'vrchol střed females',
    'pachner females',
    'angio females'
]

#SLM_COUNTS = [10, 20, 30]
SLM_COUNTS = [20]
#SLM_HANDLING = ['none', 'procd', 'bende']
SLM_HANDLING = ['none', 'procd']

IO_ERROR_SUBDIR = r'IO\ error'
OUTPUT = 'output'
#GROUP_COLORS_MAP = {'A_eneolit': (1, 0, 0),
#                    'B_bronz': (0, 1, 0),
#                    'C_latén': (0, 0, 1),
#                    'D_raný středověk': (1, 1, 0),
#                    'E_vrcholný středověk': (1, 0, 1),
#                    'F_pachner': (0, 1, 1),
#                    'G_angio': (0, 0, 0),
#                    'all': (1, 1, 1)}
GROUP_COLORS_MAP = {
    'eneolit males': (1, 0, 0),
    'bronz males': (0, 1, 0),
    'latén males': (0, 0, 1),
    'raný střed males': (1, 1, 0),
    'vrchol střed males': (1, 0, 1),
    'pachner males': (0, 1, 1),
    'angio males': (0, 0, 0),
    'eneolit females': (0.5, 0, 0),
    'bronz females': (0, 0.5, 0),
    'latén females': (0, 0, 0.5),
    'raný střed females': (0.5, 0.5, 0),
    'vrchol střed females': (0.5, 0, 0.5),
    'pachner females': (0, 0.5, 0.5),
    'angio females': (0.5, 0.5, 0.5),
    'all': (1, 1, 1)}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    return parser.parse_args()


def process_lms_handling(output_pattern, fnc, args):
    for slm in SLM_COUNTS:
        for slm_handling in SLM_HANDLING:
            fnc(output_pattern % (slm, slm_handling), OUTPUT_LOG, slm_handling, slm, args)


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_processor(output_dir, log_file, verbose=False):
    mkdir_if_not_exist(output_dir)
    if log_file and not verbose:
        sys.stdout = open(os.path.join(output_dir, log_file), 'w')
    return processcurves.CurvesProcessor(DATAFOLDER,
                                         SUBDIRS,
                                         IO_ERROR_SUBDIR,
                                         output_dir)


def get_camera_single(parallel_scale=0.4):
    return dict(position=(0, 0, 1),
                focal_point=(0, 0, 0),
                view_up=(-1, 0, 0),
                parallel_scale=parallel_scale)


def get_camera_vertical(parallel_scale=3.1):
    return [dict(position=(0, 0, 1),
                 focal_point=(0, 0, 0),
                 view_up=(1, 0, 0),
                 parallel_scale=parallel_scale),
            dict(position=(0, 1, 0),
                 focal_point=(0, 0, 0),
                 view_up=(1, 0, 0),
                 parallel_scale=parallel_scale)
            ]


def get_camera_horizontal(parallel_scale=0.4):
    return [dict(position=(0, 0, 1),
                 focal_point=(0, 0, 0),
                 view_up=(0, 1, 0),
                 parallel_scale=parallel_scale),
            dict(position=(0, 1, 0),
                 focal_point=(0, 0, 0),
                 view_up=(0, 0, 1),
                 parallel_scale=parallel_scale)
            ]
