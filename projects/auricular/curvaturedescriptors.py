#!/usr/bin/python3

""" Compute curvature descriptors on auricular shape. """

import logging
import gc
import os
import multiprocessing as mp
import numpy as np
import numpy.matlib
import numpy.linalg
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import trimesh
import trimesh.viewer

from scipy.spatial import cKDTree
from scipy import stats

from base.common import timer, runInParallel

from common import OUTPUT, DATAFOLDER, SAMPLE, DESCRIPTORS, ANALYSIS, get_sample
from report import Report
from preprocessing import render_to_file, get_mesh_data, _generate_csv
from analyze import _evaluateModel, loadData, _evaluateAllModels

from fourierdescriptors import getMaskMapping, img

PROCESSES_PER_CPU = 1

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

def removeOutliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def plotOut(fig1, filename):
    if filename:
        fig1.savefig(os.path.join(OUTPUT, filename), dpi=100)
        plt.close()
    else:
        plt.show()

def scatterPlot(x, y, filename, labels=None):
    order = np.argsort(x)
    xs = np.array(x)[order]
    ys = np.array(y)[order]

    fig1 = plt.figure()
    plt.scatter(xs, ys)
    if labels:
        for i, label in enumerate(labels):
            plt.annotate(label, (xs[i], ys[i]))
    plotOut(fig1, filename)

def boxPlot(x, filename):
    x_clean = removeOutliers(x)
    fig1 = plt.figure()
    plt.boxplot([x, x_clean])
    plotOut(fig1, filename)

def histogramPlot(x, filename):
    fig1 = plt.figure()
    plt.hist(x, bins=20)
    plotOut(fig1, filename)

def dneCurvature(vertex, neighbour_coords, vertex_normal, bandwidth):
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

def ariaDNEInternal(mesh, samples, sample_area, dist, bandwith_factor=0.4, verbose=False):
    bandwidth = dist * bandwith_factor

    normals = np.zeros(samples.shape)
    curvatures = np.zeros(samples.shape[0])

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

            curvatures[i], normals[i, :] = dneCurvature(samples[i],
                mesh.vertices[neighbours],
                mesh.vertex_normals[i],
                bandwidth)

    dne = sum(curvatures * sample_area)
    cleanDNE = sum(removeOutliers(curvatures * sample_area))
    localDNE = curvatures * sample_area

    return dict(dne=dne, localDNE=localDNE, cleanDNE=cleanDNE,
        curvature=curvatures, normals=normals, sample_area=sample_area)

@timer
def ariaDNE(mesh, dist):
    face_area = mesh.area_faces
    vert_area = [sum(face_area[faces]) / 3 for faces in mesh.vertex_faces]

    return ariaDNEInternal(mesh, mesh.vertices, vert_area, dist)

@timer
def sampledAriaDNE(mesh, dist, sample_count=5000, filter_out_borders=True):
    samples, _ = trimesh.sample.sample_surface(mesh, sample_count)

    if filter_out_borders:
        mask, mapping = getMaskMapping(mesh, 0.5, 2)

        # xxx = np.zeros(mask.shape)
        # for sample in samples:
        #     grid_coord = mapping.spaceToGrid(sample[:2])
        #     xxx[grid_coord] = xxx[grid_coord] + 1
        # img(xxx, os.path.join(OUTPUT, '_mask2.png'))

        masked_samples = [mask[mapping.spaceToGrid(sample[:2])] > 0 for sample in samples]
        samples = samples[masked_samples]

    return ariaDNEInternal(mesh, samples, np.ones(samples.shape[0]) / sample_count, dist)

def drawToFile(mesh, filename, angles=(3.14, 0, 0), distance=8):
    print("draw to file: %s" % filename)
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
    mesh = trimesh.load_mesh(specimen['filename'], process=False)
    ariadne = ariaDNE(mesh, dist=dist)
    values = np.hstack((ariadne['localDNE'], np.array([0, 0.0008])))
    colors = trimesh.visual.interpolate(np.log(1e-6 + values), 'jet')[:-2]
    output = os.path.join(OUTPUT, ariadneFilename(specimen, dist))

    mesh_data = get_mesh_data(specimen['filename'])
    mesh_data[0]['vertex_colors'] = colors
    render_to_file(output, mesh_data)

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

def computeCurvature(specimen, dist, evaluate_ariadne=False):
    if 'dist' not in specimen:
        specimen['dist'] = {}
    if dist not in specimen['dist']:
        specimen['dist'][dist] = {}
    else:
        return specimen
    print("processing %s" % specimen['basename'])
    sampled_dne = curvatureDescriptorValue(specimen, dist=dist, sampled=True)
    specimen['dist'][dist]['sampled_dne'] = sampled_dne['dne']

    if evaluate_ariadne:
        ariadne = curvatureDescriptor(specimen, dist=dist)
        specimen['dist'][dist]['ariadne'] = ariadne['dne']
        specimen['dist'][dist]['clean_ariadne'] = ariadne['cleanDNE']
        specimen['dist'][dist]['ariadne_max'] = np.max(ariadne['localDNE'])
        specimen['dist'][dist]['ariadne_local'] = ariadne['localDNE']
    return specimen

def modelEvaluation():
    results = pickle.load(open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'rb'))

    is_ariadne_evaluated = 'ariadne' in results[0]['dist'][dist]

    for result in results:
        result['sampled_dne'] = result['dist'][dist]['sampled_dne']
        if is_ariadne_evaluated:
            result['ariadne'] = result['dist'][dist]['ariadne']
            result['clean_ariadne'] = result['dist'][dist]['clean_ariadne']
        result['logAge'] = np.log(float(result['age']))

    _generate_csv(dict(output=OUTPUT, specimens=results), DESCRIPTORS,
        ('basename', 'name', 'subset', 'sex', 'age', 'side', 'logAge',
        'sampled_dne', 'ariadne', 'clean_ariadne'))
    descriptors = loadData(os.path.join(OUTPUT, DESCRIPTORS))

    if is_ariadne_evaluated:
        model_results = _evaluateAllModels(descriptors, [['sampled_dne'], ['ariadne'], ['clean_ariadne']])
    else:
        model_results = _evaluateAllModels(descriptors, [['sampled_dne']])

    analysis_result = dict(model_results=model_results)

    pickle.dump(analysis_result, open(os.path.join(OUTPUT, ANALYSIS), 'wb'))

def analyzeDescriptors(dist=0.5, upper_bound=None):
    sample = pickle.load(open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'rb'))
    sorted_subsample = sorted(sample[0:upper_bound], key=lambda spec: int(spec['age']))
    results = runInParallel([(specimen, dist) for specimen in sorted_subsample], computeCurvature, serial=False)
    pickle.dump(results, open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'wb'))

def showAnalysis(dist=0.5, upper_bound=None):
    sample = pickle.load(open(os.path.join(OUTPUT, 'curvatureDescriptor.pickle'), 'rb'))
    sorted_subsample = sorted(sample[0:upper_bound], key=lambda spec: int(spec['age']))

    is_ariadne_evaluated = 'ariadne' in sorted_subsample[0]['dist'][dist]

    table = []
    for specimen in sorted_subsample:

        if 'ariadne' in sorted_subsample[0]['dist'][dist]:
            histogram_output = "hist_%s.png" % specimen['basename']
            histogramPlot(specimen['dist'][dist]['ariadne_local'], histogram_output)

            boxplot_output = "box_%s.png" % specimen['basename']
            boxPlot(specimen['dist'][dist]['ariadne_local'], boxplot_output)

        # images = [ariadneFilename(specimen, dist), histogram_output, boxplot_output]
        images = [ariadneFilename(specimen, dist)]

        ariadne = dict(ariadne = 0, clean_ariadne=0, ariadne_max=0)
        if is_ariadne_evaluated:
            ariadne = dict(ariadne=specimen['dist'][dist]['ariadne'],
                clean_ariadne = specimen['dist'][dist]['clean_ariadne'],
                ariadne_max = specimen['dist'][dist]['ariadne_max'])

        table.append(dict(images=images,
            age=specimen['age'],
            sampled_dne=specimen['dist'][dist]['sampled_dne'],
            **ariadne,
            basename=specimen['basename']))

    ariadne_by_age = 'dne_d2_by_age_%s.png' % dist
    if is_ariadne_evaluated:
        scatterPlot([int(s['age']) for s in sorted_subsample],
            [s['dist'][dist]['ariadne'] for s in sorted_subsample],
            ariadne_by_age,
            [str(i + 1) for i in range(len(sorted_subsample))])

    sampled_dne_by_age = 'sampled_dne_by_age_%s.png' % dist
    scatterPlot([int(s['age']) for s in sorted_subsample],
        [s['dist'][dist]['sampled_dne'] for s in sorted_subsample],
        sampled_dne_by_age,
        [str(i + 1) for i in range(len(sorted_subsample))])

    analysis_result = pickle.load(open(os.path.join(OUTPUT, ANALYSIS), 'rb'))

    pdf_css = """
        body { font-size: 10px }
        img.specimen { height: 3cm }
    """
    report = Report(OUTPUT)
    report.generateCurvature(dict(table=table,
        analysis_result=analysis_result,
        ariadne_by_age=ariadne_by_age,
        sampled_dne_by_age=sampled_dne_by_age), pdf_css)

def newAnalysis():
    filename = os.path.join(OUTPUT, 'curvatureDescriptor.pickle')
    if not os.path.exists(filename):
        sample = list(get_sample(DATAFOLDER))
        pickle.dump(sample, open(filename, 'wb'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    upper_bound = None
    dist = 5.0
    newAnalysis()
    analyzeDescriptors(dist, upper_bound)
    modelEvaluation()
    showAnalysis(dist, upper_bound)
