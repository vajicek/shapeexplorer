#!/usr/bin/python3

""" Compute edge profile descriptors on auricular shape. """

import logging
import os
import gc
import numpy as np
import numpy.matlib
import numpy.linalg
import pickle

import matplotlib.pyplot as plt

import trimesh
import trimesh.viewer

import pyglet

from base.common import timer, runInParallel
from base.sampledata import get_point_cloud

from common import OUTPUT, DATAFOLDER, get_sample
from preprocessing import render_to_file, get_mesh_data
from report import Report, linePlot
from viewer import showSample

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
    p1 = points[0]
    p2 = points[-1]
    distances = []
    for p in points:
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)
    return distances


def showSamplePoints(mesh, points):
    pc = trimesh.points.PointCloud(points)
    sc = trimesh.scene.Scene([mesh, pc])
    sc.show()


@timer
def computeDescriptor(specimen, output):
    mesh = trimesh.load_mesh(specimen['filename'])
    points = getIntersections(mesh)

    if not points:
        logging.error("No profile for %s", specimen['filename'])
        return specimen

    if 'mesh_image' not in specimen:
        mesh_image_filename = os.path.join(output, specimen['basename'] + '.png')
        mesh_data = get_mesh_data(specimen['filename'])
        mesh_data += get_point_cloud(points)
        render_to_file(mesh_image_filename, mesh_data)
        specimen['mesh_image'] = mesh_image_filename

    if 'edge_profile' not in specimen:
        distances = getProfile(points)

        edge_profile_filename = specimen['basename'] + '_edge_profile.png'
        linePlot(range(len(distances)), distances, edge_profile_filename, output)
        specimen['edge_profile'] = edge_profile_filename

    gc.collect()
    return specimen


class Descriptors:

    def newAnalysis(self):
        filename = os.path.join(self.output, DESCRIPTOR_PICKLE)
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)
        if not os.path.exists(filename):
            sample = list(get_sample(DATAFOLDER))
            sample = [l for l in sample if self.subset(l)]
            pickle.dump(sample, open(filename, 'wb'))

    def analyzeDescriptors(self):
        sample = pickle.load(
            open(os.path.join(self.output, DESCRIPTOR_PICKLE), 'rb'))
        sorted_subsample = sorted(
            sample[0:self.specimen_count], key=lambda spec: int(spec['age']))
        results = runInParallel([self.descriptorInput(specimen)
                                 for specimen in sorted_subsample], computeDescriptor, serial=False)
        pickle.dump(results, open(os.path.join(
            self.output, DESCRIPTOR_PICKLE), 'wb'))


class EdgeDescriptors(Descriptors):

    def __init__(self, output, specimen_count=None, subset=lambda a: True):
        self.output = output
        self.subset = subset
        self.specimen_count = specimen_count

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

        pointCloud = trimesh.points.PointCloud(getIntersections(mesh))
        scene.delete_geometry("samplePoints")
        scene.add_geometry(pointCloud, "samplePoints")

        print(scene.current_index)
        print(scene.sample[scene.current_index])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    EdgeDescriptors(output=OUTPUT, specimen_count=50).run()

    # sorted_subsample = sorted(get_sample(DATAFOLDER), key=lambda spec: int(spec['age']))
    # print(sorted_subsample[0])
    # iii = [index for index, value in enumerate(sorted_subsample) if value['basename'] == 'LAU_47S_aur_sin_M41.ply']
    # showSample(sorted_subsample, samplePointsCallback, iii[0])
