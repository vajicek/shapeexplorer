#!/usr/bin/python3

""" Preprocess auricular shape. """

import logging
import os
import numpy as np
from base import sampledata
from base import viewer
from base.common import timer

import matplotlib.pyplot as plt

from runForAge import runForAge, runForAgeOnFiles
from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS

import trimesh
from trimesh.viewer import windowed

RESOLUTION = (1024, 1024)

old = os.path.join(DATAFOLDER, "96Cr_aur_dex_F101.ply")
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mesh = trimesh.load_mesh(old)
    heightmap = getHeightmap(mesh, 256)
    plt.imshow(heightmap)
    plt.savefig('heightmap.png')

    f1 = fftAnalysis('old_', old, [[0.55, 0.55, 0], [0.75, 0.75, 1]])
    f2 = fftAnalysis('young_', young, [[0.4, 0.4, 0], [0.6, 0.6, 1]])

    fftImg(f1 / f2, 'fft_diff.png')

    # samples = sample(mesh, 50)
    #
    # scene = trimesh.scene.Scene()
    # scene.add_geometry(trimesh.points.PointCloud(samples))
    # viewer = trimesh.viewer.windowed.SceneViewer(scene)

    # mesh = sampledata.load_ply(old)
    # print(old)
    # _view([dict(dat=mesh, col=(0.5, 0.5, 0.5))])

    #samples = sample(mesh)
    #coefs = fft(samples)
