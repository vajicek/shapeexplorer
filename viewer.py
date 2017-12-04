
import vtk

def mkVtkIdList(it):
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil

class Viewer(object):
    DEFAULT_SIZE=(300, 300)
    DEFAULT_CAMERA_POS=(1, 1, 1)
    DEFAULT_CAMERA_FP=(0, 0, 0)

    def init_gui_element(self, data):
        dataMapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            dataMapper.SetInput(data)
        else:
            dataMapper.SetInputData(data)
        dataMapper.SetScalarRange(0,7)
        dataActor = vtk.vtkActor()
        dataActor.SetMapper(dataMapper)
        return dataActor

    def set_camera(self, position=DEFAULT_CAMERA_POS, focal_point=DEFAULT_CAMERA_FP):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        self.renWin.GetRenderers().GetFirstRenderer().SetActiveCamera(camera)
        self.renWin.GetRenderers().GetFirstRenderer().ResetCamera()

    def init_window(self, data):
        renderer = vtk.vtkRenderer()
        renderer.AddActor(self.init_gui_element(data))
        renderer.SetBackground(1,1,1)

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(renderer)
        self.renWin.SetSize(*self.size)

        self.set_camera()

    def run_window(self):
        self.renWin.SetOffScreenRendering(0)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(self.renWin)
        iren.Start()

    def to_file(self, filename):
        self.renWin.SetOffScreenRendering(1)

        win2img = vtk.vtkWindowToImageFilter()
        win2img.SetInput(self.renWin)
        win2img.Update()

        pngWriter = vtk.vtkPNGWriter()
        pngWriter.SetFileName(filename)
        pngWriter.SetInputConnection(win2img.GetOutputPort())
        pngWriter.Write()

    def render(self):
        if self.filename!=None:
            self.to_file(self.filename)
        else:
            self.run_window()

    def __init__(self, data, filename=None, size=DEFAULT_SIZE):
        self.filename = filename
        self.size = size
        self.init_window(data)
