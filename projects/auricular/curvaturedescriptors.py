#!/usr/bin/python3

""" Compute curvature descriptors on auricular shape. """

import logging
import gc
import os
import multiprocessing as mp
import numpy as np
import numpy.matlib
import numpy.linalg

from base.common import timer

import matplotlib.pyplot as plt

from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS, get_sample

import trimesh
import trimesh.viewer

from scipy.spatial import cKDTree
from scipy import stats

PROCESSES_PER_CPU = 1

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def linePlot(x, y):
    order = np.argsort(x)
    xs = np.array(x)[order]
    ys = np.array(y)[order]

    plt.scatter(xs, ys)
    plt.show()

def dne_curvature(vertex, neighbour_coords, vertex_normal, bandwidth):
    i_coords = numpy.matlib.repmat(vertex, neighbour_coords.shape[0], 1)
    p = neighbour_coords - i_coords

    dist_sq = (p**2).sum(1)
    w = np.exp(-dist_sq / (bandwidth**2))

    # build covariance matrix for PCA
    C = np.zeros(6)
    C[0] = sum(p[:,0] * w * p[:,0])
    C[1] = sum(p[:,0] * w * p[:,1])
    C[2] = sum(p[:,0] * w * p[:,2])
    C[3] = sum(p[:,1] * w * p[:,1])
    C[4] = sum(p[:,1] * w * p[:,2])
    C[5] = sum(p[:,2] * w * p[:,2])
    C = C / sum(w)

    Cmat = np.array([[C[0], C[1], C[2]],
        [C[1], C[3], C[4]],
        [C[2], C[4], C[5]]])

    # compute its eigenvalues and eigenvectors
    d, v = numpy.linalg.eig(Cmat)

    # find the eigenvector that is closest to the vertex normal
    v_aug = np.hstack((v, -v))

    normal6 = numpy.matlib.repmat(np.transpose(np.array([vertex_normal])), 1, 6)
    diff = v_aug - normal6

    q = (diff**2).sum(0)

    k = np.argmin(q)

    # use that eigenvector to give an updated estimate to the vertex normal
    normal = v_aug[:,k]
    k = k % 3

    # use the eigenvalue of that egienvector to estimate the curvature
    lamb = d[k]

    sum_d = sum(d)
    curvature = 0 if sum_d == 0 else lamb / sum_d

    return curvature, normal

@timer
def ariaDNEInternal(mesh, samples, sample_normals, sample_area, dist=0.2, bandwith_factor=0.6):
    bandwidth = dist * bandwith_factor

    normals = np.zeros(sample_normals.shape)
    curvatures = np.zeros(sample_normals.shape[0])

    kdtree = cKDTree(mesh.vertices)

    batch_size = 10000
    sample_count = len(samples)
    batches = int((sample_count + batch_size) / batch_size)

    for j in range(batches):
        print("%d/%d" % (j + 1, batches))
        from_index = j * batch_size
        to_index = min((j + 1) * batch_size, sample_count)
        res = kdtree.query_ball_point(samples[from_index: to_index], r=dist)

        for sub_i, neighbours in enumerate(res):
            i = j * batch_size + sub_i
            curvatures[i], normals[i, :] = dne_curvature(samples[i],
                mesh.vertices[neighbours],
                mesh.vertex_normals[i],
                bandwidth)

    dne = sum(curvatures * sample_area)
    localDNE = curvatures * sample_area

    return dict(dne=dne, localDNE=localDNE, curvature=curvatures, normals=normals)

def ariaDNE(mesh, dist=0.2, bandwith_factor=0.6):
    face_area = mesh.area_faces
    vert_area = [sum(face_area[faces]) / 3 for faces in mesh.vertex_faces]

    return ariaDNEInternal(mesh, mesh.vertices, mesh.vertex_normals, vert_area, dist, bandwith_factor)

def sampledAriaDNE(mesh, dist=0.2, bandwith_factor=0.6, sample_count=1000):
    samples, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    sample_normals = mesh.face_normals[face_indices]
    return ariaDNEInternal(mesh, samples, sample_normals, np.ones(sample_count) / sample_count, dist, bandwith_factor)

def drawToFile(mesh, filename, angles=(3.14, 0, 0), distance=8):
    scene = trimesh.scene.Scene([mesh])
    scene.set_camera(angles=angles, distance=distance)

    png = scene.save_image(resolution=[640, 480], visible=True)
    with open(filename, 'wb') as f:
        f.write(png)
        f.close()

def curvatureDescriptorDemo():
    filename = '../ariaDNE_code/data.ply'
    mesh = trimesh.load_mesh(filename)

    for dist in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ariadne = ariaDNE(mesh, dist=dist)
        mesh.visual.vertex_colors = trimesh.visual.interpolate(np.log(0.0001 + ariadne['localDNE']), 'jet')
        drawToFile(mesh, 'output/test%s.png' % dist)

def curvatureDescriptor(filename, dists=[0.5]):
    mesh = trimesh.load_mesh(filename)
    #[0.5, 1.0, 2.0, 4.0, 8.0]
    for dist in dists:
        ariadne = ariaDNE(mesh, dist=dist)
        mesh.visual.vertex_colors = trimesh.visual.interpolate(np.log(0.0001 + ariadne['localDNE']), 'jet')
        drawToFile(mesh, 'output/arikular%s.png' % dist, angles=(0, 0, 0), distance=100)

def curvatureDescriptorValue(specimen, dist=0.5, sampled=True):
    mesh = trimesh.load_mesh(specimen['filename'])
    if sampled:
        return sampledAriaDNE(mesh, dist=dist)
    else :
        return ariaDNE(mesh, dist=dist)
    return ariadne

def sampleMesh(specimen):
    mesh = trimesh.load_mesh(specimen['filename'])
    samples, face_indices = trimesh.sample.sample_surface(mesh, 5000)
    mesh.face_normals[face_indices]
    points = trimesh.points.PointCloud(samples)
    scene = trimesh.scene.Scene([mesh, points])
    scene.show()

def descriprotAnalysis():
    sample = list(get_sample(DATAFOLDER, OUTPUT))
    #curvatureDescriptor(sample[0]['filename'])
    a = [curvatureDescriptorValue(sample[1])['dne'] for i in range(30)]
    print(stats.describe(a))
    print(a)

def analyzeDescriptors():
    sample = list(get_sample(DATAFOLDER, OUTPUT))[0:10]

    for specimen in sorted(sample, key=lambda spec: spec['age']):
        d1 = curvatureDescriptorValue(specimen, sampled=True)
        d2 = curvatureDescriptorValue(specimen, sampled=False)
        print(specimen['basename'], specimen['age'], d1['dne'], d2['dne'])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzeDescriptors()
