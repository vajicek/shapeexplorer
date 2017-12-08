
import vtk
import viewer
import math
import sampledata


def rotate_around_camera_pos(ang):
    return (math.sin(math.radians(ang)), 0, math.cos(math.radians(ang)))

def visualization_demo():
    mesh = sampledata.load_obj(
        "/home/vajicek/Dropbox/TIBIA/CURVATURE/13_IVsin_miku_EM.obj")
    for sl_count in [5,10,20,30,40]:
        sls = sampledata.load_sl_balls(
            "/home/vajicek/Dropbox/TIBIA/CURVATURE/13 IV z GOMu.asc", sl_count, 2)
        data = [] + sls
        data.append(dict(dat=mesh, col=(1, 1, 1)))
        v = viewer.Viewer(data, size=(1024, 1024))
        v.filename = "output/screenshot_sl%03d.png" % sl_count
        v.render()

def main():
    mesh = sampledata.load_obj(
        "/home/vajicek/Dropbox/TIBIA/CURVATURE/13_IVsin_miku_EM.obj")
    # curve = sampledata.load_curve(
    #    "/home/vajicek/Dropbox/TIBIA/CURVATURE/13 IV z GOMu.asc", 10)
    sls = sampledata.load_sl_balls(
        "/home/vajicek/Dropbox/TIBIA/CURVATURE/13 IV z GOMu.asc", 20, 2)
    data = [] + sls
    data.append(dict(dat=mesh, col=(1, 1, 1)))
    #data.append(dict(dat=sls, col=(1, 0, 0)))
    v = viewer.Viewer(data, size=(1024, 1024))
    v.render()
    # for i in range(0, 36):
    #    p = rotate_around_camera_pos(i * 10)
    #    #v.set_camera(p, (0,0,0))
    #    v.filename = "output/screenshot%03d.png" % i
    #    v.render()

visualization_demo()
