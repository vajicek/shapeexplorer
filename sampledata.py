import vtk
import viewer

def cube_data():
    # x = array of 8 3-tuples of float representing the vertices of a cube:
    x = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
         (0.0, 0.0, 1.0), (1.0, 0.0 ,1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]

    # pts = array of 6 4-tuples of vtkIdType (int) representing the faces
    #     of the cube in terms of the above vertices
    pts = [(0,1,2,3), (4,5,6,7), (0,1,5,4),
           (1,2,6,5), (2,3,7,6), (3,0,4,7)]

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
        scalars.InsertTuple1(i,i)

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
