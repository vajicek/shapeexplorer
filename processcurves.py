#!/usr/bin/python3

""" Process curves. """

import csv
import logging
import glob
import itertools
import os
import sampledata
import subdivcurve
import subprocess

DATAFOLDER = "/home/vajicek/Dropbox/TIBIA/CURVATURE/Tibie CURVES"
SUBDIRS = ["A_eneolit", "B_bronz", "C_latén", "D_raný středověk", "E_vrcholný středověk", "F_pachner", "G_angio"]
OUTPUTFOLDER = "output"

def load_all_curves():
    curves = {}
    for subdir in SUBDIRS:
        subdir_abs = os.path.join(DATAFOLDER, subdir)
        curves[subdir] = []
        for curve_file in glob.glob(subdir_abs + '/*.asc'):
            logging.info(curve_file)
            curve = subdivcurve.subdivide_curve(sampledata.load_polyline_data(curve_file), 30)
            curves[subdir].append(curve)
    return curves         


def call_r(script):
    cmd = ['Rscript', script]
    
    process = subprocess.Popen(' '.join(cmd), shell=True,
                           stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode('utf-8'), end='')


def store_for_r(curves):
    for category, curve_list in curves.items():
        logging.info("processing category: " + category)
        with open(os.path.join(OUTPUTFOLDER, category + '.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for curve in curve_list:
                curve_line = list(itertools.chain.from_iterable(curve))
                spamwriter.writerow([str(num) for num in curve_line])    


def analyze_curves():
    call_r('processcurves.R')


def process_curves():
    curves = load_all_curves()
    store_for_r(curves)
    analyze_curves()
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    process_curves()

