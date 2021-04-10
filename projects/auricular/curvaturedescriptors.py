#!/usr/bin/python3

""" Compute curvature descriptors on auricular shape. """

import logging
import gc
import os
import multiprocessing as mp
import numpy as np
import numpy.matlib
import numpy.linalg
import pickle

from base.common import timer

import matplotlib.pyplot as plt

from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS, get_sample

import trimesh
import trimesh.viewer

from scipy.spatial import cKDTree
from scipy import stats

from report import Report

PROCESSES_PER_CPU = 1

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def linePlot(x, y, filename):
    order = np.argsort(x)
    xs = np.array(x)[order]
    ys = np.array(y)[order]

    fig1 = plt.figure()
    plt.scatter(xs, ys)
    if filename:
        fig1.savefig(os.path.join(OUTPUT, filename), dpi=100)
        plt.close()
    else:
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
def ariaDNEInternal(mesh, samples, sample_normals, sample_area, dist=0.2, bandwith_factor=0.6, verbose=False):
    bandwidth = dist * bandwith_factor

    normals = np.zeros(sample_normals.shape)
    curvatures = np.zeros(sample_normals.shape[0])

    kdtree = cKDTree(mesh.vertices)

    batch_size = 10000
    sample_count = len(samples)
    batches = int((sample_count + batch_size) / batch_size)

    for j in range(batches):
        if verbose:
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

def ariadneFilename(specimen, dist):
    return 'ariadne_%s_%s.png' % (specimen['basename'], dist)

def curvatureDescriptor(specimen, dist=0.5):
    mesh = trimesh.load_mesh(specimen['filename'])
    ariadne = ariaDNE(mesh, dist=dist)
    values = np.hstack((ariadne['localDNE'], np.array([0, 0.0008])))
    mesh.visual.vertex_colors = trimesh.visual.interpolate(np.log(1e-6 + values), 'jet')[:-2]
    output = os.path.join(OUTPUT, ariadneFilename(specimen, dist))
    drawToFile(mesh, output, angles=(0, 0, 0), distance=100)
    return ariadne

def curvatureDescriptorValue(specimen, dist=0.5, sampled=True):
    mesh = trimesh.load_mesh(specimen['filename'])
    if sampled:
        return sampledAriaDNE(mesh, dist=dist)
    else :
        return ariaDNE(mesh, dist=dist)
    return ariadne

def sampleMeshDemo(specimen):
    mesh = trimesh.load_mesh(specimen['filename'])
    samples, face_indices = trimesh.sample.sample_surface(mesh, 5000)
    mesh.face_normals[face_indices]
    points = trimesh.points.PointCloud(samples)
    scene = trimesh.scene.Scene([mesh, points])
    scene.show()

def analyzeDescriptors(dist=0.5, slice=10):
    sample = pickle.load(open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'rb'))
    sorted_subsample = sorted(sample[0:slice], key=lambda spec: int(spec['age']))

    for specimen in sorted_subsample:
        if 'dist' not in specimen:
            specimen['dist'] = {}
        if dist not in specimen['dist']:
            specimen['dist'][dist] = {}
        else:
            continue
        print("processing %s" % specimen['basename'])
        d1 = curvatureDescriptorValue(specimen, dist=dist, sampled=True)
        d2 = curvatureDescriptor(specimen, dist=dist)
        specimen['dist'][dist]['d1.dne'] = d1['dne']
        specimen['dist'][dist]['d2.dne'] = d2['dne']
        specimen['dist'][dist]['d2.localDNE.max'] = np.max(d2['localDNE'])

        pickle.dump(sample, open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'wb'))

def showAnalysis(dist=0.5, slice=10):
    sample = pickle.load(open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'rb'))
    sorted_subsample = sorted(sample[0:slice], key=lambda spec: int(spec['age']))

    table = []
    for specimen in sorted_subsample:
        images = [ariadneFilename(specimen, dist)]
        table.append(dict(images=images,
            age=specimen['age'],
            d1_dne=specimen['dist'][dist]['d1.dne'],
            d2_dne=specimen['dist'][dist]['d2.dne'],
            d2_localDNE_max=specimen['dist'][dist]['d2.localDNE.max'],
            basename=specimen['basename']))

    dne_d2_by_age = 'dne_d2_by_age_%s.png' % dist
    linePlot([int(s['age']) for s in sorted_subsample],
        [s['dist'][dist]['d2.dne'] for s in sorted_subsample],
        'dne_d2_by_age_%s.png' % dist)

    dne_d1_by_age = 'dne_d1_by_age_%s.png' % dist
    linePlot([int(s['age']) for s in sorted_subsample],
        [s['dist'][dist]['d1.dne'] for s in sorted_subsample],
        'dne_d1_by_age_%s.png' % dist)

    pdf_css = """
        body { font-size: 10px }
        img.specimen { height: 3cm }
    """
    report = Report(OUTPUT)
    report.generateCurvature(dict(table=table,
        dne_d1_by_age=dne_d1_by_age,
        dne_d2_by_age=dne_d2_by_age), pdf_css)



def newAnalysis():
    sample = list(get_sample(DATAFOLDER))
    pickle.dump(sample, open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'wb'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #newAnalysis()
    analyzeDescriptors(0.5, 50)
    showAnalysis(0.5, 50)
