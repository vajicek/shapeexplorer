""" Viewer toll with off-screen rendering. """
import vtk


def ButtonEvent(obj, event):
    renWin = obj.GetRenderWindow()
    camera = renWin.GetRenderers().GetFirstRenderer().GetActiveCamera()
    print(camera)

        
class Viewer(object):
    """Viewer class, render off-screen or to window"""
    
    DEFAULT_SIZE = (300, 300)
    DEFAULT_CAMERA_POS = (1, 1, 1)
    DEFAULT_CAMERA_FP = (0, 0, 0)
    DEFAULT_VIEW_UP = (0, 1, 0)

    def __init__(self, data, filename=None, size=DEFAULT_SIZE):
        self.filename = filename
        self.size = size
        self.debug_cam = False
        self._init_window(data)

    def render(self):
        if self.filename != None:
            self._to_file(self.filename)
        else:
            self._run_window()

    def set_camera(self, position=DEFAULT_CAMERA_POS, focal_point=DEFAULT_CAMERA_FP, parallel_scale=0.14, view_up=DEFAULT_VIEW_UP):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*view_up)
        self.renWin.GetRenderers().GetFirstRenderer().SetActiveCamera(camera)
        self.renWin.GetRenderers().GetFirstRenderer().ResetCamera()
        camera.ParallelProjectionOn()
        camera.SetParallelScale(parallel_scale)

    def _init_gui_element(self, data):
        dataMapper = vtk.vtkPolyDataMapper()
        if isinstance(data, vtk.vtkPolyDataAlgorithm):
            dataMapper.SetInputConnection(data.GetOutputPort())
        elif isinstance(data, vtk.vtkPolyData):
            if vtk.VTK_MAJOR_VERSION <= 5:
                dataMapper.SetInput(data)
            else:
                dataMapper.SetInputData(data)
            dataMapper.SetScalarRange(0, 7)

        dataActor = vtk.vtkActor()
        dataActor.SetMapper(dataMapper)
        dataActor.GetProperty().SetPointSize(12)
        return dataActor

    def _init_window(self, data):
        """ Expect data to be vtkPolyDataAlgorithm or vtkPolyData. """
        renderer = vtk.vtkRenderer()
        for data_element in data:
            actor = self._init_gui_element(data_element["dat"])
            actor.GetProperty().SetColor(*data_element["col"]);
            renderer.AddActor(actor)
        renderer.SetBackground(1, 1, 1)

        self.renWin = vtk.vtkRenderWindow()
        if self.filename != None:
            self.renWin.SetAAFrames(8)
        self.renWin.AddRenderer(renderer)
        self.renWin.SetSize(*self.size)

        self.set_camera()

    def _run_window(self):
        self.renWin.SetOffScreenRendering(0)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(self.renWin)
        if self.debug_cam:
            iren.AddObserver("LeftButtonPressEvent", ButtonEvent)
        iren.Start()

    def _to_file(self, filename):
        self.renWin.SetOffScreenRendering(1)

        win2img = vtk.vtkWindowToImageFilter()
        win2img.SetInput(self.renWin)
        win2img.Update()

        pngWriter = vtk.vtkPNGWriter()
        pngWriter.SetFileName(filename)
        pngWriter.SetInputConnection(win2img.GetOutputPort())
        pngWriter.Write()
