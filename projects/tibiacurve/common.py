
import os
import sys

from base import processcurves

DATAFOLDER = "/home/vajicek/Dropbox/TIBIA/CURVATURE/Tibie CURVES"
SUBDIRS = ["A_eneolit", "B_bronz", "C_latén", "D_raný středověk", "E_vrcholný středověk", "F_pachner", "G_angio"]
IO_ERROR_SUBDIR = "IO error"
OUTPUT = 'output'


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_processor(output_dir, log_file):
    mkdir_if_not_exist(output_dir)
    sys.stdout = open(os.path.join(output_dir, log_file), 'w')
    return processcurves.CurvesProcessor(DATAFOLDER,
                                         SUBDIRS,
                                         IO_ERROR_SUBDIR,
                                         OUTPUT)


def get_camera():
    return [dict(position=(0, 0, 1),
                 focal_point=(0, 0, 0),
                 view_up=(0, 1, 0),
                 parallel_scale=0.04),
            dict(position=(0, -1, 0),
                 focal_point=(0, 0, 0),
                 view_up=(0, 1, 0),
                 parallel_scale=0.04)
            ]
