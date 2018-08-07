#!/usr/bin/python3

"""Load data and compute error."""

import glob
import os
import re
import sys

from base import rscriptsupport
#from projects.malakrivky import io_error


SOURCE_ROOT = os.path.expanduser('~/Dropbox/krivky_mala/clanek/GRAFY/Vstupni data_ xml a jpg a txt/pracovni pro digitalizaci/M_digitalizace krivek/')

def _LoadMorpho2DCurveData(input_dir, section="ac_inc_hard"):
    for filename in glob.glob(input_dir+'/*.txt'):
        basename = os.path.basename(filename)
        m = re.match(r".*" + section, basename)
        if m:
            print(basename)


#data = io_error._LoadMorpho2DCurveData(SOURCE_ROOT, io_error._ExtractSemilandmarksByArcCoordinates)
#print(data)
#io_error._StoreForR("soft.csv", )

_LoadMorpho2DCurveData(SOURCE_ROOT)
