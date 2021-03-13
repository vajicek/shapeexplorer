#!/usr/bin/python3

""" Preprocess auricular shape. """

import logging
import gc
import os
import multiprocessing as mp
import numpy as np

from base import sampledata
from base import viewer
from base.common import timer

import matplotlib.pyplot as plt

from runForAge import runForAge, runForAgeOnFiles
from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS

from scipy.ndimage import distance_transform_edt, binary_fill_holes

import trimesh
from trimesh.viewer import windowed

PROCESSES_PER_CPU = 1

RESOLUTION = (1024, 1024)

old = os.path.join(DATAFOLDER, "96Cr_aur_dex_F101.ply")
old2 = os.path.join(DATAFOLDER, "54Co1_aur_dex_F97.ply")
young = os.path.join(DATAFOLDER, "LAU_13S_aur_sin_M18.ply")


def _view(mesh):
    bounds = mesh[0]["dat"].GetBounds()
    scale = max(bounds[1] - bounds[0], bounds[3] -
                bounds[2], bounds[5] - bounds[4]) / 2
    v = viewer.Viewer(mesh, size=RESOLUTION)
    v.set_camera(position=(0, 0, 100), parallel_scale=scale)
    v.render()


@timer
def indicesOriginsDirections(bounds, sample_dim):

    ix = np.arange(sample_dim)
    iy = np.arange(sample_dim)

    x = np.linspace(bounds[0][0], bounds[1][0], sample_dim)
    y = np.linspace(bounds[0][1], bounds[1][1], sample_dim)

    xv, yv = np.meshgrid(x, y)
    xv = np.array(xv).flatten()
    yv = np.array(yv).flatten()
    zv = np.ones(yv.shape) * 10

    ixv, iyv = np.meshgrid(ix, iy)
    ixv = np.array(ixv).flatten()
    iyv = np.array(iyv).flatten()

    indices = list(zip(ixv, iyv))
    origins = list(zip(xv, yv, zv))
    directions = [(0, 0, -1)] * len(origins)
    return indices, origins, directions


@timer
def applyIntersections(indices, hit_coords, hm):
    for coord, index in zip(hit_coords[0], hit_coords[1]):
        array_index = indices[index]
        array_index = (hm.shape[0] - array_index[1] - 1, array_index[0])
        hm[array_index] = max(coord[2], hm[array_index])


@timer
def getHeightmap(mesh, sample_dim, subrange=[[0.0, 0.0, 0], [1.0, 1.0, 1]]):

    hm = np.ones([sample_dim, sample_dim]) * mesh.bounds[0][2]

    a = mesh.bounds[0] + subrange[0] * (mesh.bounds[1] - mesh.bounds[0])
    b = mesh.bounds[0] + subrange[1] * (mesh.bounds[1] - mesh.bounds[0])

    indices, origins, directions = indicesOriginsDirections([a, b], sample_dim)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hit_coords = intersector.intersects_location(origins, directions)

    applyIntersections(indices, hit_coords, hm)

    return hm


def fftImg(fft, filename):
    fft_img = np.fft.fftshift(np.log(np.abs(fft)))
    plt.imshow(fft_img)
    plt.savefig(filename)


def fftAnalysis(prefix, meshfile, subrange):
    mesh = trimesh.load_mesh(meshfile)

    heightmap = getHeightmap(mesh, 256, subrange)

    plt.imshow(heightmap)
    plt.savefig(prefix + 'heightmap.png')

    heightmap_fft = np.fft.fft2(heightmap)

    fftImg(heightmap_fft, prefix + 'heightmap_fft.png')

    return heightmap_fft


def fftDescriptorHeightmap(heightmap):
    return 0
    heightmap_fft = np.fft.fftshift(np.fft.fft2(heightmap))

    n = heightmap.shape[0]
    h = int(n / 4)
    heightmap_fft[2 * h, 2 * h] = 0
    low = np.sum(np.abs(heightmap_fft[h:3 * h, h:3 * h]))
    heightmap_fft[h:3 * h, h:3 * h] = 0
    high = np.sum(np.abs(heightmap_fft))
    return low / high


def patchFftDescriptor(meshfile, subrange, n=256):
    mesh = trimesh.load_mesh(meshfile)
    heightmap = getHeightmap(mesh, n, subrange)
    return fftDescriptor(heightmap)

def findPatch(meshfile):
    mesh = trimesh.load_mesh(meshfile)

    heightmap = getHeightmap(mesh, 128)

    mask = heightmap > 0

    filled = binary_fill_holes(mask)

    edt = distance_transform_edt(filled)

    a = edt.max()
    coord = np.unravel_index(edt.argmax(), edt.shape)

    return heightmap, a, coord


def extractHeightmap(heightmap, a, coord):
    a2 = int(a / 2)
    area = heightmap[(coord[0] - a2):(coord[0] + a2),
                     (coord[1] - a2):(coord[1] + a2)]
    return area


def fftDescriptorInternal(filename, no):
    logging.debug("pid=%s, no=%s, input=%s", os.getpid(), no, input)
    heightmap, a, coord = findPatch(filename)
    heightmap_area = extractHeightmap(heightmap, a, coord)
    return fftDescriptorHeightmap(heightmap_area)


def fftDescriptor(filename, no):
    fftd = fftDescriptorInternal(filename, no)
    gc.collect()
    return {'fftd': fftd}


@timer
def runFftDescriptorOnFiles(inputs):
    with mp.Pool(processes=mp.cpu_count() * PROCESSES_PER_CPU) as pool:
        async_results = [pool.apply_async(
            fftDescriptor, (input, i)) for input, i in zip(inputs, range(len(inputs)))]
        results = [async_result.get() for async_result in async_results]
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(fftDescriptor(old))
    print(fftDescriptor(old2))
    print(fftDescriptor(young))

    # fftDescriptor(old, [[0.55, 0.55, 0], [0.75, 0.75, 1]])
    # fftDescriptor(young, [[0.4, 0.4, 0], [0.6, 0.6, 1]])

    # mesh = trimesh.load_mesh(old)
    # heightmap = getHeightmap(mesh, 256)
    # plt.imshow(heightmap)
    # plt.savefig('heightmap.png')

    # f1 = fftAnalysis('old_', old, [[0.55, 0.55, 0], [0.75, 0.75, 1]])
    # f2 = fftAnalysis('young_', young, [[0.4, 0.4, 0], [0.6, 0.6, 1]])
    #
    # fftImg(f1 / f2, 'fft_diff.png')

    # samples = sample(mesh, 50)
    #
    # scene = trimesh.scene.Scene()
    # scene.add_geometry(trimesh.points.PointCloud(samples))
    # viewer = trimesh.viewer.windowed.SceneViewer(scene)

    # mesh = sampledata.load_ply(old)
    # print(old)
    # _view([dict(dat=mesh, col=(0.5, 0.5, 0.5))])

    # samples = sample(mesh)
    # coefs = fft(samples)
