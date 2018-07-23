#!/usr/bin/python3

"""Load data and compute error."""

import glob
import os
import re
import sys

from base import rscriptsupport

SOURCE_ROOT = os.path.expanduser('~/Dropbox/krivky_mala/clanek/')
TARGET_ROOT = os.path.expanduser('~/DB/krivky_mala/clanek/')
TARGET_ROOT = SOURCE_ROOT

IO_ERROR_OUTPUT_DIR = os.path.join(
    os.path.expanduser('~/DB'), 'krivky_mala/clanek/io_error/')


def _ParseCoordinates(data_block):
    coords = []
    for line in iter(data_block.splitlines()):
        coord = [float(s) for s in line.split()]
        coords.append(coord)
    return coords

def _ExtractSectionCoordinates(filename, begin_pattern, end_pattern):
    read_block = False
    data_block = ''
    with open(filename, 'r') as file:
        for line in file:
            if re.match(begin_pattern, line):
                read_block = True
            elif re.match(end_pattern, line) and read_block:
                break
            elif read_block:
                data_block += line
    return _ParseCoordinates(data_block)

def _ExtractSemilandmarksByArcCoordinates(filename):
    return _ExtractSectionCoordinates(filename,
                                      'Semilandmarks by arc',
                                      '##SECTION_END##')


def _ExtractEndPointCoordinates(filename):
    return _ExtractSectionCoordinates(filename,
                                      'Ending points',
                                      '##SECTION_END##')


def _LoadMorpho2DCurveData(input_dir,
                           method=_ExtractSemilandmarksByArcCoordinates):
    data_dict = dict()
    for filename in glob.glob(input_dir+'/*.txt'):
        print(filename)
        m = re.match(r".*[_,\/](.*)\_(.*)\_(.*)\_(.*)([0-9]+)(_opr)*\.txt",
                     filename)
        if m:
            key = m.group(1) + "_" + m.group(2) + "_" + m.group(3)
            name = m.group(4)
            repeat = int(m.group(5))
            if name not in data_dict:
                data_dict[name] = {}
            if key not in data_dict[name]:
                data_dict[name][key] = []
            coords = method(filename)
            data_dict[name][key].append(dict(
                filename=filename,
                no=repeat,
                coords=coords))
    return data_dict


def _Flatten(coords):
    return [str(coord) for lm in coords for coord in lm]


def _StoreForR(output_file, data_dict):
    #
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # key, rep, coords
    with open(output_file, 'w') as file:
        for name, value in data_dict.items():
            for key, data_n in value.items():
                for data_1 in data_n:
                    flat_coords = _Flatten(data_1["coords"])
                    file.write(",".join([name, key] + flat_coords) + "\n")


def _Analyze(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = open(os.path.join(output_dir, "results.txt"), 'w')
    bigtable_file = os.path.join(output_dir, "bigtable.csv")
    endpoints_file = os.path.join(output_dir, "endpoints.csv")
    _StoreForR(bigtable_file, _LoadMorpho2DCurveData(input_dir))
    _StoreForR(endpoints_file,
               _LoadMorpho2DCurveData(input_dir, _ExtractEndPointCoordinates))
    riface = rscriptsupport.RScripInterface(output_dir)
    riface.call_r('projects/malakrivky/io_error.R', ["--output",
                  re.escape(output_dir), "--input", re.escape(bigtable_file)])


_Analyze(os.path.join(SOURCE_ROOT,
                      '02 OPRAVA obrazky pro segmentaci/'),
         os.path.join(TARGET_ROOT,
                      'result/02 OPRAVA obrazky pro segmentaci/'))
_Analyze(os.path.join(SOURCE_ROOT,
                      '01obrazky pro segmentaci/'),
         os.path.join(TARGET_ROOT,
                      'result/01obrazky pro segmentaci/'))
