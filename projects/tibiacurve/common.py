
import os
import sys

from base import processcurves

#TARGET_ROOT = os.path.expanduser('~/Dropbox')
TARGET_ROOT = os.path.expanduser('~/DB')
DATAFOLDER = os.path.join(TARGET_ROOT, r'TIBIA/CURVATURE/Tibie\ CURVES')
#SUBDIRS = ['A_eneolit', 'B_bronz', 'C_latén', 'D_raný středověk',
#           'E_vrcholný středověk', 'F_pachner', 'G_angio']

SUBDIRS = [
    'A_eneolit/eneolit males',
    'A_eneolit/eneolit females',
    'B_bronz/bronz males',
    'B_bronz/bronz females',
    'C_latén/latén males',
    'C_latén/latén females',
    'D_raný středověk/raný střed males',
    'D_raný středověk/raný střed females',
    'E_vrcholný středověk/vrchol střed males',
    'E_vrcholný středověk/vrchol střed females',
    'F_pachner/pachner males',
    'F_pachner/pachner females',
    'G_angio/angio males',
    'G_angio/angio females'
]


IO_ERROR_SUBDIR = r'IO\ error'
OUTPUT = 'output'
GROUP_COLORS_MAP = {'A_eneolit': (1, 0, 0),
                    'B_bronz': (0, 1, 0),
                    'C_latén': (0, 0, 1),
                    'D_raný středověk': (1, 1, 0),
                    'E_vrcholný středověk': (1, 0, 1),
                    'F_pachner': (0, 1, 1),
                    'G_angio': (0, 0, 0),
                    'all': (1, 1, 1)}


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_processor(output_dir, log_file):
    mkdir_if_not_exist(output_dir)
    if log_file:
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
