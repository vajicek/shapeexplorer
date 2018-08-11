#!/usr/bin/python3

"""Load data and compute error."""

import glob
import os
import re
import sys

from base import rscriptsupport
from projects.malakrivky import io_error


#SOURCE_ROOT = os.path.expanduser('~/DB/krivky_mala/clanek/GRAFY/Vstupni data_ xml a jpg a txt/pracovni pro digitalizaci/F_digitalizace krivek2/')
SOURCE_ROOT = os.path.expanduser('./data/g_rhi')
FILE_PATTERN = r"([A-Za-z\_]+)\_(.+\_.+\_.+)\.txt"

def _GetGroups(input_dir):
    groups = {}
    for filename in glob.glob(input_dir + '/*.txt'):
        basename = os.path.basename(filename)
        m = re.match(FILE_PATTERN, basename)
        if m:
            groupname = m.group(2)
            if groupname not in groups:
                groups[groupname] = dict(count=1, files=[filename])
            else:
                groups[groupname]["count"] = groups[groupname]["count"] + 1
                groups[groupname]["files"].append(filename)
    return groups

def _LoadMorpho2DCurveData(input_dir):
    groups = _GetGroups(input_dir)
    for groupname in groups.keys():
        groups[groupname]["data"] = []
        for filename in groups[groupname]["files"]:
            coords = io_error._ExtractSemilandmarksByArcCoordinates(filename)
            groups[groupname]["data"].append(coords)
    return groups

def _Flatten(coords):
    return [str(coord) for lm in coords for coord in lm]

def _StoreForR(output_file, groups, groupname):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        i = 0
        for data1 in groups[groupname]["data"]:
            if data1[10][1] < 0:
                #print(i, groups[groupname]["files"][i], data1[10][1])
                data1=[[coord[0], -coord[1]] for coord in data1]
            i = i + 1
            flat_coords = _Flatten(data1)
            file.write(",".join(flat_coords) + "\n")


#data = io_error._LoadMorpho2DCurveData(SOURCE_ROOT, io_error._ExtractSemilandmarksByArcCoordinates)
#print(data)
#io_error._StoreForR("soft.csv", )

groups = _LoadMorpho2DCurveData(SOURCE_ROOT)
_StoreForR(os.path.expanduser('./data/koren_nosu_hard.csv'), groups, "g_rhi_hard")
_StoreForR(os.path.expanduser('./data/koren_nosu_soft.csv'), groups, "g_rhi_soft")
#print(groups.keys())
#_LoadMorpho2DCurveData(SOURCE_ROOT)
