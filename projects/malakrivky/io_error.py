#!/usr/bin/python3

"""Load data and compute error"""

import glob
import os
import re

TARGET_ROOT = os.path.expanduser('~/Dropbox')
IO_ERROR_INPUT_DIR = os.path.join(
    TARGET_ROOT,
    'krivky_mala/clanek/01obrazky pro segmentaci/')
IO_ERROR_OUTPUT_DIR = os.path.join(
    os.path.expanduser('~/DB'), 'krivky_mala/clanek/io_error/')


def _ParseCoordinates(data_block):
    coords = []
    for line in iter(data_block.splitlines()):
        coord = [float(s) for s in line.split()]
        coords.append(coord)
    return coords


def _ExtractCoordinates(filename):
    BEGIN_PATTERN = 'Semilandmarks by arc'
    END_PATTERN = '##SECTION_END##'
    read_block = False
    data_block = ''
    with open(filename, 'r') as file:
        for line in file:
            if re.match(BEGIN_PATTERN, line):
                read_block = True
            elif re.match(END_PATTERN, line) and read_block:
                break
            elif read_block:
                data_block += line
    return _ParseCoordinates(data_block)


def _LoadData(input_dir):
    data_dict = dict()
    for filename in glob.glob(input_dir+'/*.txt'):
        print(filename)
        m = re.match(r".*[_,\/](.*)\_(.*)\_(.*)\_(.*)([0-9]+)\.txt", filename)
        if m:
            key = m.group(1) + "_" + m.group(2) + "_" + m.group(3)
            name = m.group(4)
            repeat = int(m.group(5))
            if name not in data_dict:
                data_dict[name] = {}
            if key not in data_dict[name]:
                data_dict[name][key] = []
            coords = _ExtractCoordinates(filename)
            data_dict[name][key].append(dict(
                filename=filename,
                no=repeat,
                coords=coords))
    return data_dict


def _Flatten(coords):
    return [str(coord) for lm in coords for coord in lm]


def _StoreForR(output_file, data_dict):
    # key, rep, coords
    with open(output_file, 'w') as file:
        for name, value in data_dict.items():
            for key, data_n in value.items():
                for data_1 in data_n:
                    flat_coords = _Flatten(data_1["coords"])
                    file.write(",".join([name, key] + flat_coords) + "\n")


_StoreForR('bigtable.csv', _LoadData(IO_ERROR_INPUT_DIR))
