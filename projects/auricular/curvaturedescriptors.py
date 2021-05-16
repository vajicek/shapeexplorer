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

from common import OUTPUT, DATAFOLDER_SURFACE_ONLY, SAMPLE, DESCRIPTORS, ANALYSIS, get_sample
from report import Report, removeOutliers, scatterPlot, boxPlot, histogramPlot
from preprocessing import render_to_file, get_mesh_data, _generate_csv
from analyze import _evaluateModel, loadData, _evaluateAllModels

from fourierdescriptors import getMaskMapping, img, regularSampling

PROCESSES_PER_CPU = 1

DESCRIPTOR_PICKLE = 'curvatureDescriptor.pickle'

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def dneCurvature(vertex, neighbour_coords, vertex_normal, bandwidth):
    i_coords = numpy.matlib.repmat(vertex, neighbour_coords.shape[0], 1)
    p = neighbour_coords - i_coords

    dist_sq = (p**2).sum(1)
    w = np.exp(-dist_sq / (bandwidth**2))

    # build covariance matrix for PCA
    C = np.zeros(6)
    C[0] = sum(p[:, 0] * w * p[:, 0])
    C[1] = sum(p[:, 0] * w * p[:, 1])
    C[2] = sum(p[:, 0] * w * p[:, 2])
    C[3] = sum(p[:, 1] * w * p[:, 1])
    C[4] = sum(p[:, 1] * w * p[:, 2])
    C[5] = sum(p[:, 2] * w * p[:, 2])
    C = C / sum(w)

    Cmat = np.array([[C[0], C[1], C[2]],
                     [C[1], C[3], C[4]],
                     [C[2], C[4], C[5]]])

    # compute its eigenvalues and eigenvectors
    d, v = numpy.linalg.eig(Cmat)

    # find the eigenvector that is closest to the vertex normal
    v_aug = np.hstack((v, -v))

    normal6 = numpy.matlib.repmat(
        np.transpose(np.array([vertex_normal])), 1, 6)
    diff = v_aug - normal6

    q = (diff**2).sum(0)

    k = np.argmin(q)

    # use that eigenvector to give an updated estimate to the vertex normal
    normal = v_aug[:, k]
    k = k % 3

    # use the eigenvalue of that egienvector to estimate the curvature
    lamb = d[k]

    sum_d = sum(d)
    curvature = 0 if sum_d == 0 else lamb / sum_d

    return np.log(1e-6 + curvature), normal


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
        neighbourhood = kdtree.query_ball_point(samples[from_index: to_index],
                                                r=dist)

        for sub_i, neighbours in enumerate(neighbourhood):
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
def sampledAriaDNE(mesh,
                   dist=1,
                   sample_count=5000,
                   filter_out_borders=True,
                   mask_sampling_rate=0.5,
                   border_erode_iterations=0,
                   output_filename=None, **kwargs):
    # samples, _ = trimesh.sample.sample_surface_even(mesh, sample_count)

    _, _, samples = regularSampling(mesh, mask_sampling_rate)
    samples = samples[0]

    if filter_out_borders and border_erode_iterations > 0:
        mask, mapping = getMaskMapping(
            mesh, mask_sampling_rate, border_erode_iterations)

        if output_filename:
            sample_map = np.zeros(mask.shape)
            for sample in samples:
                grid_coord = mapping.spaceToGrid(sample[:2])
                sample_map[grid_coord] = sample_map[grid_coord] + 1
            img(sample_map * mask, output_filename)
            #img(mask, output_filename)

        masked_samples = [mask[mapping.spaceToGrid(sample[:2])] > 0
            for sample in samples]
        samples = samples[masked_samples]

    return ariaDNEInternal(mesh, samples, np.ones(samples.shape[0]) / samples.shape[0], dist)


def ariadneFilename(specimen, dist):
    return 'ariadne_%s_%s.png' % (specimen['basename'], dist)


def fullCurvatureDescriptorValue(specimen, dist=0.5, output_filename=None):
    mesh = trimesh.load_mesh(specimen['filename'], process=False)
    ariadne = ariaDNE(mesh, dist=dist)

    if output_filename:
        values = np.hstack((ariadne['localDNE'], np.array([0, 0.0008])))
        colors = trimesh.visual.interpolate(np.log(1e-6 + values), 'jet')[:-2]

        mesh_data = get_mesh_data(specimen['filename'])
        mesh_data[0]['vertex_colors'] = colors
        render_to_file(output_filename, mesh_data)

    return ariadne


def sampledCurvatureDescriptorValue(**kwargs):
    output = kwargs['output']
    specimen = kwargs['specimen']
    mesh_output_filename = os.path.join(output, specimen['basename'] + '_mesh.png')
    render_to_file(mesh_output_filename, get_mesh_data(specimen['filename']))

    mesh = trimesh.load_mesh(specimen['filename'])
    output_filename = os.path.join(output, specimen['basename'] + '_sample_map.png')
    return sampledAriaDNE(mesh, output_filename=output_filename, **kwargs)


def computeCurvature(**kwargs):
    gc.collect()

    specimen = kwargs['specimen']
    dist = kwargs['dist']
    output = kwargs['output']

    if 'dist' not in specimen:
        specimen['dist'] = {}
    if dist not in specimen['dist']:
        specimen['dist'][dist] = {}
    else:
        return specimen
    print("processing %s" % specimen['basename'])
    sampled_dne = sampledCurvatureDescriptorValue(**kwargs)
    specimen['dist'][dist]['sampled_dne'] = sampled_dne['dne']

    if kwargs['eval_pervertex_ariadne']:
        output_file = os.path.join(output, ariadneFilename(specimen, dist))
        ariadne = fullCurvatureDescriptorValue(specimen, dist=dist, output_file=output_file)
        specimen['dist'][dist]['ariadne'] = ariadne['dne']
        specimen['dist'][dist]['clean_ariadne'] = ariadne['cleanDNE']
        specimen['dist'][dist]['ariadne_max'] = np.max(ariadne['localDNE'])
        specimen['dist'][dist]['ariadne_local'] = ariadne['localDNE']

    return specimen


class CurvatureDescriptors:

    def __init__(self,
                 upper_bound=None,
                 dist=5.0,
                 output=OUTPUT,
                 eval_pervertex_ariadne=False,
                 subset=lambda a: True):
        self.upper_bound = upper_bound
        self.dist = dist
        self.output = output
        self.eval_pervertex_ariadne = eval_pervertex_ariadne
        self.subset = subset
        self.params = dict(dist=self.dist,
            sample_count=5000,
            filter_out_borders=True,
            mask_sampling_rate=0.5,
            border_erode_iterations=0,
            output=self.output,
            eval_pervertex_ariadne=self.eval_pervertex_ariadne)

    def newAnalysis(self):
        filename = os.path.join(self.output, DESCRIPTOR_PICKLE)
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        if not os.path.exists(filename):
            sample = list(get_sample(DATAFOLDER_SURFACE_ONLY))
            sample = [l for l in sample if self.subset(l)]
            pickle.dump(sample, open(filename, 'wb'))

    def analyzeDescriptors(self):
        sample = pickle.load(
            open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))
        sorted_subsample = sorted(
            sample[0:self.upper_bound], key=lambda spec: int(spec['age']))

        results = runInParallel([{'specimen': specimen, **self.params}
                                 for specimen in sorted_subsample], computeCurvature, serial=False)
        pickle.dump(results, open(os.path.join(
            self.output, DESCRIPTOR_PICKLE), 'wb'))

    def modelEvaluation(self):
        results = pickle.load(
            open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))

        for result in results:
            result['sampled_dne'] = result['dist'][self.dist]['sampled_dne']
            if self.eval_pervertex_ariadne:
                result['ariadne'] = result['dist'][self.dist]['ariadne']
                result['clean_ariadne'] = result['dist'][self.dist]['clean_ariadne']
            result['logAge'] = np.log(float(result['age']))
            result['subsets'] = ['all']

        _generate_csv(dict(output=self.output, specimens=results), DESCRIPTORS,
                      ('basename', 'name', 'subset', 'sex', 'age', 'side', 'logAge',
                       'sampled_dne', 'ariadne', 'clean_ariadne'))
        descriptors = loadData(os.path.join(self.output, DESCRIPTORS))

        if self.eval_pervertex_ariadne:
            model_results = _evaluateAllModels(descriptors,
                                               [['sampled_dne'], ['ariadne'], ['clean_ariadne']])
        else:
            model_results = _evaluateAllModels(descriptors,
                                               [['sampled_dne']])

        analysis_result = dict(model_results=model_results)

        pickle.dump(analysis_result, open(
            os.path.join(self.output, ANALYSIS), 'wb'))

    def showAnalysis(self):
        sample = pickle.load(
            open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))
        sorted_subsample = sorted(
            sample[0:self.upper_bound], key=lambda spec: int(spec['age']))

        table = []
        for specimen in sorted_subsample:

            if 'ariadne' in sorted_subsample[0]['dist'][self.dist]:
                histogram_output = "hist_%s.png" % specimen['basename']
                histogramPlot(
                    specimen['dist'][self.dist]['ariadne_local'], histogram_output, self.output)

                boxplot_output = "box_%s.png" % specimen['basename']
                boxPlot(specimen['dist'][self.dist]
                        ['ariadne_local'], boxplot_output, self.output)

            sample_map = specimen['basename'] + '_sample_map.png'
            mesh_output = specimen['basename'] + '_mesh.png'

            # images = [ariadneFilename(specimen, dist), histogram_output, boxplot_output]
            images = [ariadneFilename(specimen, self.dist), sample_map, mesh_output]

            ariadne = dict(ariadne=0, clean_ariadne=0, ariadne_max=0)
            if self.eval_pervertex_ariadne:
                ariadne = dict(ariadne=specimen['dist'][self.dist]['ariadne'],
                               clean_ariadne=specimen['dist'][self.dist]['clean_ariadne'],
                               ariadne_max=specimen['dist'][self.dist]['ariadne_max'])

            table.append(dict(images=images,
                              age=specimen['age'],
                              sampled_dne=specimen['dist'][self.dist]['sampled_dne'],
                              **ariadne,
                              basename=specimen['basename']))

        labels = [str(i + 1) for i in range(len(sorted_subsample))]

        ariadne_by_age = 'dne_d2_by_age_%s.png' % self.dist
        if self.eval_pervertex_ariadne:
            scatterPlot([int(s['age']) for s in sorted_subsample],
                        [s['dist'][self.dist]['ariadne']
                            for s in sorted_subsample],
                        ariadne_by_age,
                        self.output,
                        labels)

        sampled_dne_by_age = 'sampled_dne_by_age_%s.png' % self.dist
        scatterPlot([int(s['age']) for s in sorted_subsample],
                    [s['dist'][self.dist]['sampled_dne']
                        for s in sorted_subsample],
                    sampled_dne_by_age,
                    self.output,
                    labels)

        analysis_result = pickle.load(
            open(os.path.join(self.output, ANALYSIS), 'rb'))

        pdf_css = """
            body { font-size: 10px }
            img.specimen { height: 3cm }
        """
        report = Report(self.output)
        report.generateCurvature(dict(table=table,
                                      params=self.params,
                                      eval_pervertex_ariadne=self.eval_pervertex_ariadne,
                                      analysis_result=analysis_result,
                                      ariadne_by_age=ariadne_by_age,
                                      sampled_dne_by_age=sampled_dne_by_age), pdf_css)

    def run(self):
        self.newAnalysis()
        self.analyzeDescriptors()
        self.modelEvaluation()
        self.showAnalysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # CurvatureDescriptors(upper_bound=None, dist=5.0, eval_pervertex_ariadne=False,
    #                      output=OUTPUT + "_lt_60",
    #                      subset=lambda a: int(a['age']) < 60).run()
    #
    # CurvatureDescriptors(upper_bound=None, dist=5.0, eval_pervertex_ariadne=False,
    #                      output=OUTPUT + "_ge_60",
    #                      subset=lambda a: int(a['age']) >= 60).run()

    CurvatureDescriptors(upper_bound=None,
                         dist=1.0,
                         eval_pervertex_ariadne=False,
                         output=OUTPUT).run()
