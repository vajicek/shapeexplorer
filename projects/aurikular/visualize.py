#!/usr/bin/python3

""" Visualize and analyze auricular shape. """

import glob
import logging
import math
import os
import re
from base import sampledata
from base import viewer
from base.common import timer
from runForAge import runForAge, runForAgeOnFiles

OUTPUT = "../output"
DATAFOLDER = "~/data/aurikularni_plocha_ply/"
RESOLUTION = (1024, 1024)
FILENAME_PATTERN = re.compile(r'.*/(.*)_(aur)_(dex|sin)_(F|M)([0-9]*)')

def get_mesh_data(filename):
    mesh = sampledata.load_ply(filename)
    return [dict(dat=mesh, col=(0.5, 0.5, 0.5))]

def render_to_file(filename, mesh):
    bounds = mesh[0]["dat"].GetBounds()
    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2
    v = viewer.Viewer(mesh, size=RESOLUTION)
    v.filename = os.path.join(filename)
    v.set_camera(position=(0, 0, 100), parallel_scale=scale)
    v.render()

def parse_name(filename):
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {'basename': os.path.basename(filename),
        'filename': filename,
        'name': match.group(1),
        'type': match.group(2),
        'side': match.group(3),
        'sex': match.group(4),
        'age': match.group(5)
    }

@timer
def get_sample(input_folder, output):
    ply_files_glob = os.path.join(os.path.expanduser(input_folder), "*.ply")
    specimens = []
    for abs_filename in glob.glob(ply_files_glob):
        specimen = parse_name(abs_filename)
        if not specimen:
            logging.error("Failed to parse filename: %s", abs_filename)
        else:
            specimens.append(specimen)
    return {'specimens': specimens, 'output': output}

@timer
def generate_images(input_sample, force=False):
    sample = input_sample.copy()
    for specimen in sample['specimens']:
        png_filename = os.path.splitext(specimen['basename'])[0] + ".png"
        output_filename = os.path.join(sample["output"], png_filename)
        specimen['output'] = output_filename
        if not os.path.exists(output_filename) or force:
            mesh = get_mesh_data(specimen['filename'])
            render_to_file(output_filename, mesh)
    return sample

@timer
def generate_image_grid(sample):
    with open(os.path.join(sample['output'], 'index.html'), 'w') as html:
        for specimen in sample['specimens']:
            html.write('%s, %s, %s, %s, %s<br/>\n' % (specimen['name'],
                specimen['sex'],
                specimen['age'],
                specimen['side'],
                specimen['basename']))
            relative_path = os.path.relpath(specimen['output'], sample['output'])
            html.write('<img src="%s"/>\n' % relative_path)
            html.write('<br/>\n')

@timer
def generate_csv(sample, filename, columns):
    with open(os.path.join(sample['output'], filename), 'w') as csv:
        csv.write('%s\n' % ",".join(columns))
        for specimen in sample['specimens']:
            values = [str(specimen.get(column, '')) for column in columns]
            csv.write('%s\n' % ",".join(values))

@timer
def run_forAge_on_sample(input_sample):
    sample = input_sample.copy()
    results = runForAgeOnFiles([specimen['filename'] for specimen in sample['specimens']])
    for result, specimen in zip(results, sample['specimens']):
        specimen.update(result)
    return sample

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample = get_sample(DATAFOLDER, OUTPUT)
    sample = generate_images(sample)
    generate_image_grid(sample)
    generate_csv(sample, 'sample.csv', ('name', 'sex', 'age', 'side', 'basename'))
    sample = run_forAge_on_sample(sample)
    generate_csv(sample, 'sample_descriptors.csv', ('name', 'sex', 'age', 'side', 'BE', 'SAH', 'VC'))
