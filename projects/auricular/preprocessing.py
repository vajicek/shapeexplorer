""" Preprocess auricular shape. """

import os

import matplotlib.pyplot as plt

from base import sampledata
from base import viewer
from base.common import timer

RESOLUTION = (1024, 1024)


def img(array2d, filename, colorbar=False):
    plt.imshow(array2d)
    if colorbar:
        plt.colorbar()
    plt.savefig(filename)
    plt.close()


def getMeshData(filename):
    mesh = sampledata.load_ply(filename)
    return [dict(dat=mesh, col=(0.5, 0.5, 0.5))]


def renderToFile(filename, mesh):
    bounds = mesh[0]["dat"].GetBounds()
    scale = max(bounds[1] - bounds[0], bounds[3] -
                bounds[2], bounds[5] - bounds[4]) / 2
    viewer_instance = viewer.Viewer(mesh, size=RESOLUTION)
    viewer_instance.filename = os.path.join(filename)
    viewer_instance.set_camera(position=(0, 0, 100), parallel_scale=scale)
    viewer_instance.render()


@timer
def generateImages(input_sample, force=False):
    sample = input_sample.copy()
    for specimen in sample['specimens']:
        png_filename = os.path.splitext(specimen['basename'])[0] + ".png"
        output_filename = os.path.join(sample["output"], png_filename)
        specimen['output'] = output_filename
        if not os.path.exists(output_filename) or force:
            mesh = getMeshData(specimen['filename'])
            renderToFile(output_filename, mesh)
    return sample


def generateCsv(sample, filename, columns):
    with open(os.path.join(sample['output'], filename), 'w') as csv:
        csv.write('%s\n' % ",".join(columns))
        for specimen in sample['specimens']:
            values = [str(specimen.get(column, '')) for column in columns]
            csv.write('%s\n' % ",".join(values))
