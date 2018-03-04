#!/usr/bin/python3

""" Test loading curves. """

import logging
import os
import processcurves
import sampledata
import viewer

RESOLUTION = (1024, 1024)
OUTPUTFOLDER = "output"


def plot_curves(curves):
    data = []
    for group in curves.values():
        for curve in group:
            curve_balls = sampledata.create_balls(curve, 1, color=(1, 0, 0))
            data = data + curve_balls 
    v = viewer.Viewer(data, size=RESOLUTION)
    v.filename = os.path.join(OUTPUTFOLDER, "all_points.png")
    v.render()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    plot_curves(processcurves.load_all_curves())

