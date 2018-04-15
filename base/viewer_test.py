#!/usr/bin/python3

""" Test for viewer. """
import math
import os
from base import sampledata
from base import viewer

DATAFOLDER = "testdata"
OUTPUTFOLDER = "output"
RESOLUTION = (1024, 1024)


def rotate_around_camera_pos(ang):
    return (math.sin(math.radians(ang)), 0, math.cos(math.radians(ang)))


def render(viewer):
    viewer.set_camera(position=(0, 0, 1), parallel_scale=200)
    viewer.render()


def visualization_generator_demo():
    mesh = sampledata.load_obj(os.path.join(DATAFOLDER, "13_IVsin_miku_EM.obj"))
    for sl_count in [5, 10, 20, 30, 40]:
        sls = sampledata.load_sl_balls(os.path.join(DATAFOLDER, "13 IV z GOMu.asc"), sl_count, 2)
        v = viewer.Viewer(sls + [dict(dat=mesh, col=(1, 1, 1))], size=RESOLUTION)
        v.filename = os.path.join(OUTPUTFOLDER, "screenshot_sl%03d.png" % sl_count)
        render(v)


def viewer_demo():
    mesh = sampledata.load_obj(os.path.join(DATAFOLDER, "13_IVsin_miku_EM.obj"))
    ls = sampledata.load_sl_balls(os.path.join(DATAFOLDER, "13 IV z GOMu.asc"), None, 1, (0, 1, 0))
    sls = sampledata.load_sl_balls(os.path.join(DATAFOLDER, "13 IV z GOMu.asc"), 30, 2)
    data = ls + sls + [dict(dat=mesh, col=(1, 1, 1))]
    v = viewer.Viewer(data, size=RESOLUTION)
    render(v)


if __name__ == "__main__":
    visualization_generator_demo()
    viewer_demo()

