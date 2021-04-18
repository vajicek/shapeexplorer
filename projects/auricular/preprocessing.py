#!/usr/bin/python3

""" Preprocess auricular shape. """

import logging
import math
import os
from base import sampledata
from base import viewer
from base.common import timer

from runForAge import runForAge, runForAgeOnFiles
from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS, get_sample
from fourierdescriptors import runFftDescriptorOnFiles

RESOLUTION = (1024, 1024)

# ERROR:root:Failed to parse filename: /home/vajicek/data/aurikularni_plocha_ply/466co2_aur_dex_M44.ply
# ERROR:root:Failed to parse filename: /home/vajicek/data/aurikularni_plocha_ply/114co2_aur_sin_M32.ply
# ERROR:root:Failed to parse filename: /home/vajicek/data/aurikularni_plocha_ply/804-58_4811th_aur_sin_F37.ply


def get_mesh_data(filename):
    mesh = sampledata.load_ply(filename)
    return [dict(dat=mesh, col=(0.5, 0.5, 0.5))]


def render_to_file(filename, mesh):
    bounds = mesh[0]["dat"].GetBounds()
    scale = max(bounds[1] - bounds[0], bounds[3] -
                bounds[2], bounds[5] - bounds[4]) / 2
    v = viewer.Viewer(mesh, size=RESOLUTION)
    v.filename = os.path.join(filename)
    v.set_camera(position=(0, 0, 100), parallel_scale=scale)
    v.render()


def _get_sample(input_folder, output):
    specimens = []
    for specimen in get_sample(input_folder, output):
        if not specimen:
            logging.error("Failed to parse filename: %s", abs_filename)
        else:
            specimens.append(specimen)
    return {'specimens': specimens, 'output': output}


@timer
def _generate_images(input_sample, force=False):
    sample = input_sample.copy()
    for specimen in sample['specimens']:
        png_filename = os.path.splitext(specimen['basename'])[0] + ".png"
        output_filename = os.path.join(sample["output"], png_filename)
        specimen['output'] = output_filename
        if not os.path.exists(output_filename) or force:
            mesh = get_mesh_data(specimen['filename'])
            render_to_file(output_filename, mesh)
    return sample


def _generate_csv(sample, filename, columns):
    with open(os.path.join(sample['output'], filename), 'w') as csv:
        csv.write('%s\n' % ",".join(columns))
        for specimen in sample['specimens']:
            values = [str(specimen.get(column, '')) for column in columns]
            csv.write('%s\n' % ",".join(values))


@timer
def _run_forAge_on_sample(input_sample):
    sample = input_sample.copy()
    results = runForAgeOnFiles([specimen['filename']
                                for specimen in sample['specimens']])
    for result, specimen in zip(results, sample['specimens']):
        specimen.update(result)
    return sample


@timer
def _run_fftdescriptors_on_sample(input_sample):
    sample = input_sample.copy()
    results = runFftDescriptorOnFiles([specimen['filename']
                                       for specimen in sample['specimens'][0:10]])
    for result, specimen in zip(results, sample['specimens']):
        specimen.update(result)
    return sample

# import matplotlib.pyplot as plt
# import pandas as pd
#
# def _genAgeDescriptorPlots(folder, dataframe):
#     plots=[]
#     for x in ['fftd']:
#         filename = 'scatter_' + x + '.png'
#         output_filepath = os.path.join(folder, filename)
#         plots.append({'filename': filename})
#         dataframe.plot.scatter(x='age', y=x)
#         plt.savefig(output_filepath, dpi=100)
#     return plots


@timer
def preprocessing(input, output):
    sample = _get_sample(input, output)
    sample = _generate_images(sample)
    _generate_csv(sample, SAMPLE, ('name', 'subset',
                                   'sex', 'age', 'side', 'basename'))
    sample = _run_fftdescriptors_on_sample(sample)
    #sample = _run_forAge_on_sample(sample)
    _generate_csv(sample, DESCRIPTORS, ('basename', 'name',
                                        'subset', 'sex', 'age', 'side', 'BE', 'SAH', 'VC', 'fftd'))

    # frame = pd.read_csv(os.path.join(sample['output'], DESCRIPTORS), sep=',', quotechar='"')
    # _genAgeDescriptorPlots(sample['output'], frame)
    # print(frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    preprocessing(DATAFOLDER, OUTPUT)
