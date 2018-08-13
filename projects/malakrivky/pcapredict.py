#!/usr/bin/python3

"""Load data and compute error."""

import glob
import os
import re
import sys

from base import rscriptsupport
from projects.malakrivky import io_error


#PARTS = ['koren_nosu', 'hrbet_nosu', 'horni_ret', 'dolni_ret']
PARTS = ['hrbet_nosu']
DATA_ROOT = os.path.expanduser('./data')
#FILE_PATTERN = r"([A-Za-z\_]+)\_(.+\_.+\_.+)\.txt"
FILE_PATTERN = r"([A-Za-z\_]+)\_(.+)\.txt"
OUTPUT_DIR = '/home/vajicek/Dropbox/krivky_mala/clanek/GRAFY/predikce/'

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
                data1=[[coord[0], -coord[1]] for coord in data1]
            i = i + 1
            flat_coords = _Flatten(data1)
            file.write(",".join(flat_coords) + "\n")

def _ProccessWithR(output_dir, part_name):
    riface = rscriptsupport.RScripInterface(output_dir)
    riface.call_r('projects/malakrivky/pcapredict.R', [
        "--output", re.escape(output_dir),
        "--part", re.escape(part_name)])

def main():
    for part_name in PARTS:
        groups = _LoadMorpho2DCurveData(os.path.join(DATA_ROOT, part_name))
        hard_key = [name for name in groups.keys() if 'hard' in name][0]
        soft_key = [name for name in groups.keys() if 'soft' in name][0]
        _StoreForR(os.path.join(DATA_ROOT, part_name + "_hard.csv"), groups, hard_key)
        _StoreForR(os.path.join(DATA_ROOT, part_name + "_soft.csv"), groups, soft_key)
        _ProccessWithR(OUTPUT_DIR, part_name)

if __name__ == "__main__":
    main()
