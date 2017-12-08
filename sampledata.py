import vtk
import viewer
import subdivcurve


def cube_data():
    # x = array of 8 3-tuples of float representing the vertices of a cube:
    x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]

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
        points.InsertPoint(i, x[i])
    for i in range(6):
        polys.InsertNextCell(viewer.mkVtkIdList(pts[i]))
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


def curve(polyline_data):
    points = vtk.vtkPoints()
    for i, point in enumerate(polyline_data):
        points.InsertPoint(i, *point)

    linesource = vtk.vtkLineSource()
    linesource.SetPoints(points)

    tubefilter = vtk.vtkTubeFilter()
    tubefilter.SetInputConnection(linesource.GetOutputPort())
    tubefilter.SetRadius(1)
    tubefilter.SetNumberOfSides(50)
    tubefilter.Update()
    return tubefilter


def load_polyline_data(polyline_data_file):
    polyline_data = []
    with open(polyline_data_file, "r") as f:
        for line in f:
            tokens = line.split(" ")
            vector = [float(a) for a in tokens[1:4]]
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


def load_sl_balls(curve_file, sl_count, radius):
    polyline_data = load_polyline_data(curve_file)
    sls = subdivcurve.subdivide_curve(polyline_data, sl_count)
    balls = []
    for i, sl in enumerate(sls):
        ball = vtk.vtkSphereSource()
        ball.SetCenter(*sl)
        ball.SetRadius(radius)
        balls.append(dict(dat=ball, col=(1, 0, 0)))
    return balls

def load_curve(curve_file, subdivide_to=None):
    polyline_data=load_polyline_data(curve_file)
    if subdivide_to:
        return curve(subdivcurve.subdivide_curve(polyline_data, subdivide_to))
    else:
        return curve(polyline_data)
