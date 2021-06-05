#!/usr/bin/python3

""" Compute curvature descriptors on auricular shape. """

import logging
import gc
import os
import pickle

from typing import Callable
from typing import NamedTuple

import numpy as np
import numpy.matlib
import numpy.linalg

import trimesh
import trimesh.viewer

import scipy.spatial

from base.common import timer, runInParallel

from .common import OUTPUT, DATAFOLDER_SURFACE_ONLY, DESCRIPTORS, ANALYSIS, get_sample
from .report import Report, removeOutliers, scatterPlot, boxPlot, histogramPlot
from .preprocessing import render_to_file, get_mesh_data, generate_csv, img
from .analyze import loadData, evaluateAllModels

from .projection import getMaskMapping, regularSampling, getMapping

PROCESSES_PER_CPU = 1

DESCRIPTOR_PICKLE = 'curvatureDescriptor.pickle'

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def estimateCurvature(cov_mat, vertex_normal):
    # compute its eigenvalues and eigenvectors
    eigenvalues, eigenvectors = numpy.linalg.eig(cov_mat)

    # find the eigenvector that is closest to the vertex normal
    v_aug = np.hstack((eigenvectors, -eigenvectors))

    normal6 = numpy.matlib.repmat(
        np.transpose(np.array([vertex_normal])), 1, 6)
    diff = v_aug - normal6

    sq_distances = (diff**2).sum(0)

    closes_vector_index = np.argmin(sq_distances)

    # use that eigenvector to give an updated estimate to the vertex normal
    normal = v_aug[:, closes_vector_index]
    closes_vector_index = closes_vector_index % 3

    # use the eigenvalue of that egienvector to estimate the curvature
    lamb = eigenvalues[closes_vector_index]

    sum_eigenvalues = sum(eigenvalues)
    curvature = 0 if sum_eigenvalues == 0 else lamb / sum_eigenvalues

    return curvature, normal


def dneCurvature(vertex, neighbour_coords, vertex_normal, bandwidth):
    vertex_coords = numpy.matlib.repmat(vertex, neighbour_coords.shape[0], 1)
    centered = neighbour_coords - vertex_coords

    dist_sq = (centered**2).sum(1)
    weights = np.exp(-dist_sq / (bandwidth**2))

    # build covariance matrix for PCA
    covs = np.zeros(6)
    covs[0] = sum(centered[:, 0] * weights * centered[:, 0])
    covs[1] = sum(centered[:, 0] * weights * centered[:, 1])
    covs[2] = sum(centered[:, 0] * weights * centered[:, 2])
    covs[3] = sum(centered[:, 1] * weights * centered[:, 1])
    covs[4] = sum(centered[:, 1] * weights * centered[:, 2])
    covs[5] = sum(centered[:, 2] * weights * centered[:, 2])
    covs = covs / sum(weights)

    cov_mat = np.array([[covs[0], covs[1], covs[2]],
                        [covs[1], covs[3], covs[4]],
                        [covs[2], covs[4], covs[5]]])

    return estimateCurvature(cov_mat, vertex_normal)


def ariaDneInternal(mesh, samples, sample_normals, sample_area,
                    dist, bandwith_factor=0.4, verbose=False, **_):
    bandwidth = dist * bandwith_factor

    normals = np.zeros(samples.shape)
    curvatures = np.zeros(samples.shape[0])

    kdtree = scipy.spatial.cKDTree(mesh.vertices)  # pylint: disable=no-member

    batch_size = 10000
    sample_count = len(samples)
    batches = int((sample_count + batch_size) / batch_size)

    for j in range(batches):
        if verbose:
            print("%d/%d" % (j + 1, batches))
        from_index = j * batch_size
        to_index = min((j + 1) * batch_size, sample_count)
        neighbourhood = kdtree.query_ball_point(samples[from_index: to_index], r=dist)

        for sub_i, neighbours in enumerate(neighbourhood):
            i = j * batch_size + sub_i

            curvatures[i], normals[i, :] = dneCurvature(samples[i],
                                                        mesh.vertices[neighbours],
                                                        sample_normals[i],
                                                        bandwidth)

    dne = sum(curvatures * sample_area)
    clean_dne = sum(removeOutliers(curvatures * sample_area))
    local_dne = curvatures * sample_area

    return dict(dne=dne, localDNE=local_dne, cleanDNE=clean_dne, samples=samples,
                curvature=curvatures, normals=normals, sample_area=sample_area)


def getSamples(mesh, sample_count, sampling_rate, sampling_method, **_):
    if sampling_method == 'trimesh_even':
        samples, face_index = trimesh.sample.sample_surface_even(mesh, sample_count)
        sample_normals = mesh.face_normals[face_index]
    elif sampling_method == 'regular':
        _, _, samples, sample_normals = regularSampling(mesh, sampling_rate)
    return samples, sample_normals


def filterOutSamples(samples,
                     sample_normals,
                     mesh,
                     filter_out_borders,
                     border_erode_iterations,
                     mask_sampling_rate,
                     **_):
    if filter_out_borders and border_erode_iterations > 0:
        mask = getMaskMapping(mesh, mask_sampling_rate, border_erode_iterations)
        mapping = getMapping(mesh, mask_sampling_rate)
        masked_samples = [mask[mapping.spaceToGrid(sample[:2])] > 0 for sample in samples]
        samples = samples[masked_samples]
        sample_normals = sample_normals[masked_samples]
    return samples, sample_normals


@timer
def sampledAriaDne(mesh, **kwargs):
    samples, sample_normals = getSamples(mesh, **kwargs)
    samples, sample_normals = filterOutSamples(samples, sample_normals, mesh, **kwargs)
    return ariaDneInternal(mesh=mesh,
                           samples=samples,
                           sample_normals=sample_normals,
                           sample_area=np.ones(samples.shape[0]) / samples.shape[0],
                           **kwargs)


def ariadneFilename(specimen, dist):
    return 'ariadne_%s_%s.png' % (specimen['basename'], dist)


@timer
def ariaDne(mesh, dist):
    face_area = mesh.area_faces
    vert_area = [sum(face_area[faces]) / 3 for faces in mesh.vertex_faces]
    return ariaDneInternal(mesh, mesh.vertices, mesh.vertex_normals, vert_area, dist)


def fullCurvatureDescriptorValue(specimen, dist=0.5, output_filename=None):
    mesh = trimesh.load_mesh(specimen['filename'], process=False)
    ariadne = ariaDne(mesh, dist=dist)

    if output_filename:
        values = np.hstack((ariadne['localDNE'], np.array([0, 0.0008])))
        colors = trimesh.visual.interpolate(np.log(1e-6 + values), 'jet')[:-2]

        mesh_data = get_mesh_data(specimen['filename'])
        mesh_data[0]['vertex_colors'] = colors
        render_to_file(output_filename, mesh_data)

    return ariadne


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
    print("processing %s (%d/%d)" % (specimen['basename'], kwargs['specimen_no'], kwargs['specimen_total']))

    mesh = trimesh.load_mesh(specimen['filename'])

    sampled_dne = sampledAriaDne(mesh=mesh, **kwargs)
    specimen['dist'][dist]['sampled_dne'] = sampled_dne['dne']
    specimen['dist'][dist]['curvature'] = sampled_dne['curvature']

    # output sample map
    sample_map_output_filename = os.path.join(output, specimen['basename'] + '_sample_map.png')
    mapping = getMapping(mesh, kwargs['sampling_rate'])
    sample_map = np.zeros(np.flip(mapping.grid_dim)) + np.log(1e-6)
    for sample, curvature in zip(sampled_dne['samples'], sampled_dne['curvature']):
        grid_coord = mapping.spaceToGrid(sample[:2])
        #sample_map[grid_coord] = sample_map[grid_coord] + 1
        sample_map[grid_coord] = np.log(1e-6 + curvature)
    img(sample_map, sample_map_output_filename, colorbar=True)

    if kwargs['eval_pervertex_ariadne']:
        output_file = os.path.join(output, ariadneFilename(specimen, dist))
        ariadne = fullCurvatureDescriptorValue(specimen, dist=dist, output_filename=output_file)
        specimen['dist'][dist]['ariadne'] = ariadne['dne']
        specimen['dist'][dist]['clean_ariadne'] = ariadne['cleanDNE']
        specimen['dist'][dist]['ariadne_max'] = np.max(ariadne['localDNE'])
        specimen['dist'][dist]['ariadne_local'] = ariadne['localDNE']

    return specimen


def positive(numbers):
    return numbers[numbers > 1e-9]

def rejectOutliers(data, mean=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

class HistogramDescriptors:

    def __init__(self, data, dist):
        self.data = data
        self.dist = dist

    def _curvature(self, specimen):
        return positive(self.data[specimen]['dist'][self.dist]['curvature'])

    def _minmax(self):
        specimens = len(self.data)
        list_of_arrays = list(self._curvature(i) for i in range(specimens))
        mins = [np.min(l) for l in list_of_arrays]
        maxs = [np.max(l) for l in list_of_arrays]
        return min(mins), max(maxs)

    def getHistogramData(self, i, bins, minmin, maxmax):
        log_curvature = np.log(self._curvature(i))
        hist, _ = np.histogram(log_curvature,
            bins=bins,
            range=(np.log(minmin), np.log(maxmax)))
        return hist / np.sum(hist)

    def getSampleHistogramData(self, bins):
        minmin, maxmax = self._minmax()
        data_array = np.ndarray((len(self.data), bins))
        for i in range(len(self.data)):
            data_array[i] = self.getHistogramData(i, bins, minmin, maxmax)
        return data_array


class CurvatureDescriptorsParams(NamedTuple):
    input_data: str = DATAFOLDER_SURFACE_ONLY
    upper_bound: int = None
    dist: float = 5.0
    output: str = OUTPUT
    eval_pervertex_ariadne: bool = False
    subset: Callable[[dict], bool] = lambda a: True
    sampling_rate: float = 0.2
    filter_out_borders: bool = False
    sampling_method: str = 'regular'
    sample_count: int = 5000
    border_erode_iterations: int = 0
    mask_sampling_rate: float = 0.5


class CurvatureDescriptors:

    def __init__(self, params):
        self.params = params

    def newAnalysis(self):
        filename = os.path.join(self.params.output, DESCRIPTOR_PICKLE)
        if not os.path.exists(self.params.output):
            os.makedirs(self.params.output, exist_ok=True)
        if not os.path.exists(filename):
            sample = list(get_sample(self.params.input_data))
            sample = [l for l in sample if self.params.subset(l)]
            self.persistData(sample)

    def getData(self):
        return pickle.load(open(os.path.join(self.params.output, DESCRIPTOR_PICKLE), 'rb'))

    def persistData(self, results):
        pickle.dump(results, open(os.path.join(self.params.output, DESCRIPTOR_PICKLE), 'wb'))

    def _getParamDict(self):
        param_dict = dict(self.params._asdict())
        del param_dict['subset']
        return param_dict

    @timer(level=logging.INFO)
    def computeDescriptors(self):
        sample = self.getData()
        sorted_subsample = sorted(sample[0:self.params.upper_bound], key=lambda spec: int(spec['age']))

        results = runInParallel([{
                'specimen': specimen,
                'specimen_no': specimen_no,
                'specimen_total': len(sorted_subsample),
                **self._getParamDict()}
            for specimen_no, specimen in enumerate(sorted_subsample)],
            computeCurvature, serial=False)
        self.persistData(results)

    def modelEvaluation(self):
        results = self.getData()

        for result in results:
            result['sampled_dne'] = result['dist'][self.params.dist]['sampled_dne']
            if self.params.eval_pervertex_ariadne:
                result['ariadne'] = result['dist'][self.params.dist]['ariadne']
                result['clean_ariadne'] = result['dist'][self.params.dist]['clean_ariadne']
            result['logAge'] = np.log(float(result['age']))
            result['subsets'] = ['all']

        generate_csv(dict(output=self.params.output, specimens=results), DESCRIPTORS,
                     ('basename', 'name', 'subset', 'sex', 'age', 'side', 'logAge',
                      'sampled_dne', 'ariadne', 'clean_ariadne'))
        descriptors = loadData(os.path.join(self.params.output, DESCRIPTORS))

        models = [['sampled_dne']]
        if self.params.eval_pervertex_ariadne:
            models += [['ariadne'], ['clean_ariadne']]
        model_results = evaluateAllModels(descriptors, models, dep=['logAge'])

        pickle.dump(dict(model_results=model_results), open(os.path.join(self.params.output, ANALYSIS), 'wb'))

    def renderMeshImages(self):
        sample = self.getData()
        for specimen in sample:
            mesh_output_filename = os.path.join(self.params.output, specimen['basename'] + '_mesh.png')
            render_to_file(mesh_output_filename, get_mesh_data(specimen['filename']))

    def showAnalysis(self):
        sample = self.getData()
        sorted_subsample = sorted(sample[0:self.params.upper_bound], key=lambda spec: int(spec['age']))
        self._generatePdf(sorted_subsample)

    def _generateScatterplots(self, sorted_subsample):
        labels = [str(i + 1) for i in range(len(sorted_subsample))]

        ariadne_by_age = 'dne_d2_by_age_%s.png' % self.params.dist
        if self.params.eval_pervertex_ariadne:
            scatterPlot([int(s['age']) for s in sorted_subsample],
                        [s['dist'][self.params.dist]['ariadne'] for s in sorted_subsample],
                        ariadne_by_age,
                        self.params.output,
                        labels)

        sampled_dne_by_age = 'sampled_dne_by_age_%s.png' % self.params.dist
        scatterPlot([int(s['age']) for s in sorted_subsample],
                    [s['dist'][self.params.dist]['sampled_dne'] for s in sorted_subsample],
                    sampled_dne_by_age,
                    self.params.output,
                    labels)

        return ariadne_by_age, sampled_dne_by_age

    def _getSpecimenTableData(self, sorted_subsample):
        table = []
        for specimen in sorted_subsample:

            if 'ariadne' in sorted_subsample[0]['dist'][self.params.dist]:
                histogram_output = "hist_%s.png" % specimen['basename']
                histogramPlot(specimen['dist'][self.params.dist]['ariadne_local'],
                    histogram_output, self.params.output)

                boxplot_output = "box_%s.png" % specimen['basename']
                boxPlot(specimen['dist'][self.params.dist]['ariadne_local'],
                    boxplot_output, self.params.output)

            histogram_output = "hist_%s.png" % specimen['basename']
            histogramPlot(np.log(specimen['dist'][self.params.dist]['curvature']),
                histogram_output, self.params.output)

            sample_map = specimen['basename'] + '_sample_map.png'
            mesh_output = specimen['basename'] + '_mesh.png'

            # images = [ariadneFilename(specimen, dist), histogram_output, boxplot_output]
            images = []
            if self.params.eval_pervertex_ariadne:
                images = [ariadneFilename(specimen, self.params.dist)]
            images += [sample_map, mesh_output, histogram_output]

            ariadne = dict(ariadne=0, clean_ariadne=0, ariadne_max=0)
            if self.params.eval_pervertex_ariadne:
                ariadne = dict(ariadne=specimen['dist'][self.params.dist]['ariadne'],
                               clean_ariadne=specimen['dist'][self.params.dist]['clean_ariadne'],
                               ariadne_max=specimen['dist'][self.params.dist]['ariadne_max'])

            table.append(dict(images=images,
                              age=specimen['age'],
                              sampled_dne=specimen['dist'][self.params.dist]['sampled_dne'],
                              **ariadne,
                              basename=specimen['basename']))
        return table

    def _generatePdf(self, sorted_subsample):

        ariadne_by_age, sampled_dne_by_age = self._generateScatterplots(sorted_subsample)

        table = self._getSpecimenTableData(sorted_subsample)

        analysis_result = pickle.load(open(os.path.join(self.params.output, ANALYSIS), 'rb'))

        pdf_css = """
            body { font-size: 10px }
            img.specimen { height: 3cm }
        """
        report = Report(self.params.output)
        report.generateCurvature(dict(table=table,
                                      params=self.params,
                                      analysis_result=analysis_result,
                                      ariadne_by_age=ariadne_by_age,
                                      sampled_dne_by_age=sampled_dne_by_age), pdf_css)

    def run(self):
        self.newAnalysis()
        self.computeDescriptors()
        self.modelEvaluation()
        self.renderMeshImages()
        self.showAnalysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    CurvatureDescriptors(CurvatureDescriptorsParams(
        upper_bound=None,
        sampling_method='regular',
        dist=2.0,
        sampling_rate=0.5,
        sample_count=5000,
        output=OUTPUT)).run()
