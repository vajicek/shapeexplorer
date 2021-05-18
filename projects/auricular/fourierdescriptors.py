#!/usr/bin/python3

""" Compute fourier descriptors on auricular shape. """

import logging
import gc
import os
import numpy as np

import matplotlib.pyplot as plt

from base.common import timer, runInParallel

from .common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS, get_sample
from .report import Report
from .projection import computeHeightmap
from .preprocessing import img, generate_csv, generate_images

from scipy.ndimage import distance_transform_edt, binary_fill_holes

from rasterio.fill import fillnodata

import trimesh

PROCESSES_PER_CPU = 1

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def fftImg(fft, filename):
    fft_img = np.fft.fftshift(np.log(np.abs(fft)))
    plt.imshow(fft_img)
    plt.savefig(filename)
    plt.close()


def get1dFft(heightmap_fft_abs):
    n = heightmap_fft_abs.shape[0]
    h = int(n / 2)
    oned_fft = np.zeros(n)
    for x in range(heightmap_fft_abs.shape[1]):
        for y in range(heightmap_fft_abs.shape[0]):
                d = int(np.sqrt((x - h)**2 + (y - h)**2))
                if d < n:
                    oned_fft[d] += heightmap_fft_abs[x, y]
    return oned_fft


def fftDescriptorHeightmap(heightmap, f=0.5, dbg=False, i=0):
    fft = np.fft.fft2(heightmap)
    fft[0, 0] = 0
    heightmap_fft = np.fft.fftshift(fft)
    heightmap_fft_abs = np.abs(heightmap_fft)

    n = heightmap.shape[0]

    heightmap_fft_abs_normalized = heightmap_fft_abs

    oned_fft = get1dFft(heightmap_fft_abs)

    low = np.sum(oned_fft[0:int(n/4)])
    high = np.sum(oned_fft[int(n/4):int(n/2)])

    if dbg:
        fig1 = plt.figure()
        plt.ylim((0, 3500))
        plt.plot(range(oned_fft.shape[0]-1), oned_fft[1:])
        fig1.savefig(os.path.join(OUTPUT, '_1dfft_%d.png' % i))
        plt.close()

    return high / low, low, high, oned_fft


def getHeightmap(filename, sampling_resolution):
    mesh = trimesh.load_mesh(filename)
    heightmap, _ = computeHeightmap(mesh, sampling_resolution)
    return heightmap


def findPatch(heightmap, dbg=False):
    mask = heightmap > 0

    filled = binary_fill_holes(mask)

    filled_pixels = filled.astype(np.float32) - mask.astype(np.float32)

    if dbg:
        img(filled_pixels, os.path.join(OUTPUT, '_pixels_to_interpolate.png'))
        img(heightmap, os.path.join(OUTPUT, '_heightmap.png'))

    interpolated_holes = fillnodata(heightmap, mask=filled_pixels == 0)

    if dbg:
        img(interpolated_holes, os.path.join(OUTPUT, '_interpolated_holes.png'))

    edt = distance_transform_edt(filled)

    if dbg:
        img(edt, os.path.join(OUTPUT, '_edt.png'))

    patch_size = edt.max() * np.sqrt(2)
    coord = np.unravel_index(edt.argmax(), edt.shape)

    return patch_size, coord


def extractSubImage(heightmap, a, coord):
    a2 = int(a / 2)
    area = np.copy(heightmap[(coord[0] - a2):(coord[0] + a2),
                             (coord[1] - a2):(coord[1] + a2)])
    heightmap[(coord[0] - a2):(coord[0] + a2),
              (coord[1] - a2):(coord[1] + a2)] += 1
    return area


def fftDescriptor(filename, no=0, dbg=False, sampling_resolution=1, common_patch_size=1):
    _logger.debug("pid=%s, no=%s, input=%s", os.getpid(), no, filename)
    heightmap = getHeightmap(filename, sampling_resolution)

    patch_size, coord = findPatch(heightmap, dbg)

    heightmap_area = extractSubImage(heightmap, common_patch_size, coord)

    img(heightmap, os.path.join(OUTPUT, 'fft',
                                os.path.basename(filename) + '_area.png'))


    fftd, low, high, oned_fft = fftDescriptorHeightmap(heightmap_area, dbg=dbg, i=no)
    gc.collect()
    return {'fftd': fftd, 'low': low, 'high': high, '1d': oned_fft}


@timer
def runFftDescriptorOnFilesParallel(inputs, sampling_resolution=1, common_patch_size=1, proc_per_cpu=4):
    input_params = [(input['filename'], i, i==0, sampling_resolution, common_patch_size)
        for input, i in zip(inputs, range(len(inputs)))]
    results = runInParallel(input_params, fftDescriptor)
    return list(zip(inputs, results))


def getBounds(filename):
    return trimesh.load_mesh(filename).bounds


@timer
def getSamplingResolution(inputs, sampling_rate, proc_per_cpu=8):
    bounds = runInParallel([(s['filename'],) for s in inputs], getBounds)

    maxx = np.max([b[1][0] - b[0][0] for b in bounds])
    maxy = np.max([b[1][1] - b[0][1] for b in bounds])
    max_dim = np.max([maxx, maxy])
    sampling_resolution = max_dim / sampling_rate
    _logger.debug("sampling_resolution=%s", sampling_resolution)
    return sampling_resolution


def getPatch(filename, sampling_resolution):
    print(filename, sampling_resolution)
    return findPatch(getHeightmap(filename, sampling_resolution))


@timer
def getCommonSamplePatchSize(inputs, sampling_resolution, proc_per_cpu=8):
    patches = runInParallel([(s['filename'], sampling_resolution) for s in inputs], getBounds)
    common_patch_size = np.min([patch[0] for patch in patches])
    _logger.debug("common_patch_size=%s", common_patch_size)
    return common_patch_size


@timer
def _run_fftdescriptors_on_sample(input_sample):
    sample = input_sample.copy()
    results = runFftDescriptorOnFiles([specimen['filename']
                                       for specimen in sample['specimens'][0:10]])
    for result, specimen in zip(results, sample['specimens']):
        specimen.update(result)
    return sample


@timer
def preprocessing(input, output):
    sample = get_sample(input)
    sample = generate_images(sample)
    generate_csv(sample, SAMPLE, ('name', 'subset',
                                   'sex', 'age', 'side', 'basename'))
    sample = _run_fftdescriptors_on_sample(sample)
    #sample = _run_forAge_on_sample(sample)
    generate_csv(sample, DESCRIPTORS, ('basename', 'name',
                                       'subset', 'sex', 'age', 'side', 'BE', 'SAH', 'VC', 'fftd'))

    # frame = pd.read_csv(os.path.join(sample['output'], DESCRIPTORS), sep=',', quotechar='"')
    # _genAgeDescriptorPlots(sample['output'], frame)
    # print(frame)


def runFftDescriptorOnFiles(inputs, sampling_rate=192):
    sampling_resolution = getSamplingResolution(inputs, sampling_rate)
    common_patch_size = getCommonSamplePatchSize(inputs, sampling_resolution)
    files_descriptors = runFftDescriptorOnFilesParallel(inputs, sampling_resolution, common_patch_size)

    return files_descriptors

def renderReport():
    report = Report(OUTPUT)
    report.generateFft()


def analyzeDescriptors():
    sample = list(get_sample(DATAFOLDER))[0:100]
    fftd = []
    ages = []

    files_descriptors = runFftDescriptorOnFiles(sample)

    fftd = [fd[1]['fftd'] for fd in files_descriptors]
    ages = [int(fd[0]['age']) for fd in files_descriptors]

    plt.close()
    fig1 = plt.figure()
    plt.scatter(x=ages, y=fftd)
    fig1.savefig(os.path.join(OUTPUT, "_age_fftd.png"), dpi=100)
    plt.close()

    fig1 = plt.figure()
    maxnamelen = max(len(fd[0]['filename']) for fd in files_descriptors)
    for i, fd in enumerate(files_descriptors):
        print(("%4d %" + str(maxnamelen + 4) + "s %5s %.4f") % (i,
            fd[0]['filename'], fd[0]['age'], fd[1]['fftd']))
        l = fd[1]['1d']
        color = 'blue'
        if int(fd[0]['age']) > 50:
            color = 'red'
        plt.plot(range(l.shape[0]-1), l[1:], color=color)
    fig1.savefig(os.path.join(OUTPUT, '_1dfft.png'))
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessing(DATAFOLDER, OUTPUT)
    #analyzeDescriptors()
    renderReport()
    # sample = list(get_sample(DATAFOLDER, OUTPUT))[0:2]
    # runFftDescriptorOnFiles(sample)
