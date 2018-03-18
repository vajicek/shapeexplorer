""" Sample data for viewer. """

import logging
import vtk

from base import subdivcurve


def _make_vtk_id_list(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def cube_gizmo_data():
    # x = array of 8 3-tuples of float representing the vertices of a cube:
    #x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    #     (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
    x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.6, 0.0), (0.0, 0.6, 0.0),
         (0.0, 0.0, 0.3), (1.0, 0.0, 0.3), (1.0, 0.6, 0.3), (0.0, 0.6, 0.3)]

    # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
    #     of the cube in terms of the above vertices
    pts = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]

    # We'll create the building blocks of polydata including data attributes.
    cube = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()

    # Load the point, cell, and data attributes.
    for i in range(8):
        points.InsertPoint(i, [y*0.1 for y in x[i]])
    for i in range(6):
        polys.InsertNextCell(_make_vtk_id_list(pts[i]))
    for i in range(8):
        scalars.InsertTuple1(i, i)

    # We now assign the pieces to the vtkPolyData.
    cube.SetPoints(points)
    del points
    cube.SetPolys(polys)
    del polys
    cube.GetPointData().SetScalars(scalars)
    del scalars

    return cube


def load_obj(filename):
    obj_reader = vtk.vtkOBJReader()
    obj_reader.SetFileName(filename)
    obj_reader.Update()
    return obj_reader.GetOutput()


def curve(polyline_data, radius=1):
    points = vtk.vtkPoints()
    for i, point in enumerate(polyline_data):
        points.InsertPoint(i, *point)

    linesource = vtk.vtkLineSource()
    linesource.SetPoints(points)

    tubefilter = vtk.vtkTubeFilter()
    tubefilter.SetInputConnection(linesource.GetOutputPort())
    tubefilter.SetRadius(radius)
    tubefilter.SetNumberOfSides(50)
    tubefilter.Update()
    return tubefilter


def load_polyline_data(polyline_data_file):
    polyline_data = []
    with open(polyline_data_file, "r") as f:
        for line in f:
            tokens = line.split(" ")
            vector = [float(a) for a in tokens[1:4]]

            if len(polyline_data) > 0:
                dist_a_b = subdivcurve.dist(polyline_data[-1], vector)
                if dist_a_b < 1e-9:
                    logging.debug("subsequent points are two close: " + str(polyline_data[-1]) + " - " + str(vector))
                    continue
            
            polyline_data.append(vector)
    return polyline_data


def load_sl(curve_file, sl_count):
    polyline_data = load_polyline_data(curve_file)
    sls = subdivcurve.subdivide_curve(polyline_data, sl_count)
    points = vtk.vtkPoints()
    for i, sl in enumerate(sls):
        points.InsertPoint(i, *sl)

    pointspolydata = vtk.vtkPolyData()
    pointspolydata.SetPoints(points)

    vertexglyphfilter = vtk.vtkVertexGlyphFilter()
    vertexglyphfilter.SetInputData(pointspolydata)
    vertexglyphfilter.Update()

    polydata = vtk.vtkPolyData()
    polydata.ShallowCopy(vertexglyphfilter.GetOutput())
    return polydata


def create_balls(points, radius, color=(1, 0, 0), values=None):
    balls = []
    if values:
        min_values = min(values)
        max_values = max(values)
    for idx, sl in enumerate(points):
        ball = vtk.vtkSphereSource()
        ball.SetCenter(*sl)
        ball.SetThetaResolution(64)
        ball.SetPhiResolution(64)
        if values:
            r01 = (values[idx] - min_values) / (max_values - min_values)
            r = r01 * (radius[1] - radius[0]) + radius[0]  
        else:
            r = radius
        ball.SetRadius(r)
        balls.append(dict(dat=ball, col=color))
    return balls


def load_sl_balls(curve_file, sl_count, radius, color=(1, 0, 0)):
    points = load_polyline_data(curve_file)
    if sl_count:
        points = subdivcurve.subdivide_curve(points, sl_count)
    return create_balls(points, radius, color)


def load_curve(curve_file, subdivide_to=None):
    polyline_data = load_polyline_data(curve_file)
    if subdivide_to:
        return curve(subdivcurve.subdivide_curve(polyline_data, subdivide_to))
    else:
        return curve(polyline_data)
