#!/usr/bin/python3

""" Test loading curves. """

import logging
import os
import sys
from base import processcurves
from base import sampledata
from base import viewer

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
    plot_curves(processcurves.CurvesProcessor(sys.argv[1], [sys.argv[2]], None)._load_all_curves(20))

