#!/usr/bin/python3

""" Compute edge profile descriptors on auricular shape. """

import logging
import os
import gc
import pickle
from abc import ABC, abstractmethod

import numpy as np
import trimesh
import trimesh.viewer

from base.common import timer, runInParallel
from base.sampledata import get_point_cloud

from .common import OUTPUT, DATAFOLDER, getSample
from .preprocessing import renderToFile, getMeshData
from .report import Report, linePlot

DESCRIPTOR_PICKLE = 'edgeDescriptor.pickle'

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def getIntersections(mesh):
    mesh_dim = mesh.bounds[1] - mesh.bounds[0]
    center = mesh.bounds[0] + mesh_dim * np.array([0.5, 0.75, 1])
    direction = np.array([1, 1, 0])
    mesh_bb_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    origins = []
    samples = 200
    for i in range(samples):
        origin = center + (i / samples) * mesh_bb_size * direction
        origins += [origin]

    directions = [(0, 0, -1)] * len(origins)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    hit_coords = intersector.intersects_location(origins, directions)

    return [x for x, _ in sorted(zip(hit_coords[0], hit_coords[1]), key=lambda pair: pair[1])]


def getProfile(points):
    first = points[0]
    last = points[-1]
    distances = []
    for point in points:
        dist = np.linalg.norm(np.cross(last - first, first - point)) / np.linalg.norm(last - first)
        distances.append(dist)
    return distances


def showSamplePoints(mesh, points):
    point_cloud = trimesh.points.PointCloud(points)
    scene = trimesh.scene.Scene([mesh, point_cloud])
    scene.show()


@timer
def computeDescriptor(specimen, output):
    mesh = trimesh.load_mesh(specimen['filename'])
    points = getIntersections(mesh)

    if not points:
        logging.error("No profile for %s", specimen['filename'])
        return specimen

    if 'mesh_image' not in specimen:
        mesh_image_filename = os.path.join(output, specimen['basename'] + '.png')
        mesh_data = getMeshData(specimen['filename'])
        mesh_data += get_point_cloud(points)
        renderToFile(mesh_image_filename, mesh_data)
        specimen['mesh_image'] = mesh_image_filename

    if 'edge_profile' not in specimen:
        distances = getProfile(points)

        edge_profile_filename = specimen['basename'] + '_edge_profile.png'
        linePlot(range(len(distances)), distances, edge_profile_filename, output)
        specimen['edge_profile'] = edge_profile_filename

    gc.collect()
    return specimen


class Descriptors(ABC):

    def __init__(self, output, specimen_count, subset):
        self.output = output
        self.specimen_count = specimen_count
        self.subset = subset

    def newAnalysis(self):
        filename = os.path.join(self.output, DESCRIPTOR_PICKLE)
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        if not os.path.exists(filename):
            sample = list(getSample(DATAFOLDER))
            sample = [l for l in sample if self.subset(l)]
            pickle.dump(sample, open(filename, 'wb'))

    def analyzeDescriptors(self):
        sample = pickle.load(open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))
        sorted_subsample = sorted(
            sample[0:self.specimen_count],
            key=lambda spec: int(spec['age']))
        results = runInParallel(
            [self.descriptorInput(specimen) for specimen in sorted_subsample],
            computeDescriptor,
            serial=False)
        pickle.dump(results, open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'wb'))

    @abstractmethod
    def descriptorInput(self, specimen):
        ...


class EdgeDescriptors(Descriptors):

    def __init__(self, output, specimen_count=None, subset=lambda a: True):
        super().__init__(output, specimen_count, subset)

    def descriptorInput(self, specimen):
        return (specimen, self.output)

    def showAnalysis(self):
        sample = pickle.load(
            open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))
        sorted_subsample = sorted(
            sample[0:self.specimen_count], key=lambda spec: int(spec['age']))

        table = []
        for specimen in sorted_subsample:
            images = []
            if 'edge_profile' in specimen:
                images += [specimen['edge_profile']]
            if 'mesh_image' in specimen:
                images += [specimen['mesh_image']]
            table.append(dict(images=images,
                              age=specimen['age'],
                              basename=specimen['basename']))

        pdf_css = """
            body { font-size: 10px }
            img.specimen { height: 3cm }
        """
        report = Report(self.output)
        report.generateEdgeProfile(dict(table=table), pdf_css)

    def run(self):
        self.newAnalysis()
        self.analyzeDescriptors()
        # self.modelEvaluation()
        self.showAnalysis()


def samplePointsCallback(scene):
    if scene.new_index != scene.current_index:
        scene.current_index = scene.new_index

        mesh = trimesh.load_mesh(scene.sample[scene.current_index]['filename'])
        scene.delete_geometry("specimen")
        scene.add_geometry(mesh, "specimen")

        point_cloud = trimesh.points.PointCloud(getIntersections(mesh))
        scene.delete_geometry("samplePoints")
        scene.add_geometry(point_cloud, "samplePoints")

        print(scene.current_index)
        print(scene.sample[scene.current_index])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    EdgeDescriptors(output=OUTPUT, specimen_count=50).run()

    # sorted_subsample = sorted(getSample(DATAFOLDER), key=lambda spec: int(spec['age']))
    # print(sorted_subsample[0])
    # iii = [index for index, value in enumerate(sorted_subsample) if value['basename'] == 'LAU_47S_aur_sin_M41.ply']
    # showSample(sorted_subsample, samplePointsCallback, iii[0])
