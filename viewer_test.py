
import vtk
import viewer
import math
import sampledata

def rotate_around_camera_pos(ang):
    return (math.sin(math.radians(ang)), 0, math.cos(math.radians(ang)))

def main():
    data = sampledata.load_obj("/home/vajicek/Dropbox/TIBIA/CURVATURE/13_IVsin_miku_EM.obj")
    #data = sampledata.cube_data()

    v = viewer.Viewer(data, size=(1024,1024))
    for i in range(0,36):
        p=rotate_around_camera_pos(i*10)
        v.set_camera(p, (0,0,0))
        v.filename = "screenshot%03d.png" % i
        v.render()

main()
