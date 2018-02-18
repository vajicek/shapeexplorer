#!/usr/bin/python3

""" Process curves. """

import logging
import glob
import os
import sampledata
import subdivcurve
import re
import rscriptsupport
import viewer

# analysis parameters
SEMILANDMARKS = 30
DATAFOLDER = "/home/vajicek/Dropbox/TIBIA/CURVATURE/Tibie CURVES"
SUBDIRS = ["A_eneolit", "B_bronz", "C_latén", "D_raný středověk", "E_vrcholný středověk", "F_pachner", "G_angio"]
IOERROR_SUBDIR = "IO error"


def load_curves_in_dir(subdir, curves):
    subdir_abs = os.path.join(DATAFOLDER, subdir)
    curves[subdir] = []
    for curve_file in glob.glob(subdir_abs + '/*.asc'):
        logging.info(curve_file)
        curve = subdivcurve.subdivide_curve(sampledata.load_polyline_data(curve_file), SEMILANDMARKS)
        curves[subdir].append(curve)
    return curves


def load_all_curves():
    curves = {}
    for subdir in SUBDIRS:
        curves = load_curves_in_dir(subdir, curves)
    return curves         


def load_io_error_curves():
    subdir_abs = os.path.join(DATAFOLDER, IOERROR_SUBDIR)
    curves={}
    for curve_file in glob.glob(subdir_abs + '/*.asc'):
        logging.info(curve_file)
        m = re.search(".*\/(.*)\ (\d+)\..*$", curve_file)
        if m:      
            specimen_name = m.groups()[0]
            curve = subdivcurve.subdivide_curve(sampledata.load_polyline_data(curve_file), SEMILANDMARKS)
            if specimen_name not in curves:
                curves[specimen_name] = []
            curves[specimen_name].append(curve)
    return curves   


def analyze_curves():
    rscriptsupport.call_r('processcurves.R')


def show_curves(data, filename=None, radius=0.001, res=(1024, 1024)):
    vdata = []
    for unused_key, group in data.items():
        for curve in group:
            sls = sampledata.create_balls(curve, radius, color=(1, 0, 0))
            vdata = vdata + sls
    v = viewer.Viewer(vdata, size=res)
    v.filename = filename
    v.render()


def visualize_all():
    data = rscriptsupport.load_from_r("output/all_gpa.csv")
    groups = rscriptsupport.load_csv("output/all_group.csv")
    for i in range(len(data[""])):
        tmpdata = {"": data[""][0:(i + 1)]}
        show_curves(tmpdata, "output/filename%04d_%s.png" % (i, groups[i][0]))


def visualize_means():
    means = rscriptsupport.load_from_r("output/mean.csv")
    groups = rscriptsupport.load_csv("output/mean_group.csv")
    data = {}
    for i in range(len(means[""])):
        data[groups[i][0]] = means[""][0:(i + 1)]
    show_curves(data)


def process_curves():
    if rscriptsupport.curve_files_uptodate():
        curves = load_all_curves()
        rscriptsupport.store_for_r(curves)
    if rscriptsupport.curve_files_uptodate('io_error'):
        rscriptsupport.store_for_r(load_io_error_curves(), prefix='io_error')
    analyze_curves()
    #visualize_all()
    #visualize_means()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    process_curves()

