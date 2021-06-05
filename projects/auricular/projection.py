""" Projection of 3D shape to plane/regular space. """

import logging
import numpy as np

import trimesh

from scipy.ndimage import binary_erosion

from base.common import timer

_logger = logging.getLogger(__name__)


@timer
def indicesOriginsDirections(bounds, dims):

    x_range = np.arange(dims[0])
    y_range = np.arange(dims[1])

    x_space = np.linspace(bounds[0][0], bounds[1][0], dims[0])
    y_space = np.linspace(bounds[0][1], bounds[1][1], dims[1])

    x_grid, y_grid = np.meshgrid(x_space, y_space)
    x_v = np.array(x_grid).flatten()
    y_v = np.array(y_grid).flatten()
    z_v = np.ones(y_v.shape) * 10

    ixv, iyv = np.meshgrid(x_range, y_range)
    ixv = np.array(ixv).flatten()
    iyv = np.array(iyv).flatten()

    indices = list(zip(ixv, iyv))
    origins = list(zip(x_v, y_v, z_v))
    directions = [(0, 0, -1)] * len(origins)
    return indices, origins, directions


class Mapping:
    def __init__(self, from_coord, to_coord, sampling_resolution, grid_dim):
        self.from_coord = from_coord[:2]
        self.to_coord = to_coord[:2]
        self.step = sampling_resolution
        self.grid_dim = grid_dim

    def spaceToGrid(self, point):
        index3 = (point - self.from_coord) / self.step
        return (self.grid_dim[1] - int(index3[1]) - 1, int(index3[0]))


def regularSampling(mesh, sampling_resolution, subrange=None):
    subrange = subrange or [[0.0, 0.0, 0], [1.0, 1.0, 1]]

    dims = np.ceil((mesh.bounds[1] - mesh.bounds[0]) / sampling_resolution)[:2].astype(int)

    from_coord = mesh.bounds[0] + subrange[0] * (mesh.bounds[1] - mesh.bounds[0])
    to_coord = mesh.bounds[0] + subrange[1] * (mesh.bounds[1] - mesh.bounds[0])

    indices, origins, directions = indicesOriginsDirections([from_coord, to_coord], dims)

    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    coords, _, face_index = intersector.intersects_location(origins, directions)

    sample_normals = mesh.face_normals[face_index]

    return dims, indices, coords, sample_normals


def getMapping(mesh, sampling_resolution, subrange=None):
    subrange = subrange or [[0.0, 0.0, 0], [1.0, 1.0, 1]]

    dims, _, _, _ = regularSampling(mesh, sampling_resolution, subrange)
    return Mapping(mesh.bounds[0], mesh.bounds[1], sampling_resolution, dims)


@timer
def applyIntersections(indices, coords, heightmap):
    for coord, index in zip(coords[0], coords[1]):
        array_index = indices[index]
        array_index = (heightmap.shape[0] - array_index[1] - 1, array_index[0])
        heightmap[array_index] = max(coord[2], heightmap[array_index])


@timer
def computeHeightmap(mesh, sampling_resolution, subrange=None):
    subrange = subrange or [[0.0, 0.0, 0], [1.0, 1.0, 1]]

    dims, indices, coords, _ = regularSampling(mesh, sampling_resolution, subrange)
    _logger.debug("dims=%s", dims)

    heightmap = np.ones(np.flip(dims)) * mesh.bounds[0][2]
    applyIntersections(indices, coords, heightmap)

    return heightmap


def getMaskMapping(mesh, sampling_resolution, erode_by=1):
    heightmap = computeHeightmap(mesh, sampling_resolution)
    mask = heightmap > mesh.bounds[0][2]
    if erode_by > 0:
        mask = binary_erosion(mask, iterations=erode_by)
    return mask
