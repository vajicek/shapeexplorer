#!/usr/bin/python3

""" Preprocess auricular shape. """

import os

import matplotlib.pyplot as plt

from base import sampledata
from base import viewer
from base.common import timer

from .runForAge import runForAgeOnFiles

RESOLUTION = (1024, 1024)


def img(array2d, filename, colorbar=False):
    plt.imshow(array2d)
    if colorbar:
        plt.colorbar()
    plt.savefig(filename)
    plt.close()


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


def generate_csv(sample, filename, columns):
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
