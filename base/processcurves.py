#!/usr/bin/python3

""" Process curves. """

import logging
import glob
import os
import re

from base import sampledata
from base import subdivcurve
from base import rscriptsupport
from base import viewer


class CurvesProcessor(object):

    def __init__(self, datafolder, subdirs, io_error_subdir, output):
        self.datafolder = datafolder
        self.subdirs = subdirs
        self.io_error_subdir = io_error_subdir
        self.riface = rscriptsupport.RScripInterface(output)
    
    def _load_curves_in_dir(self, subdir, curves, semilandmarks):
        subdir_abs = os.path.join(self.datafolder, subdir)
        curves[subdir] = []
        for curve_file in glob.glob(subdir_abs + '/*.asc'):
            logging.info(curve_file)
            curve = subdivcurve.subdivide_curve(sampledata.load_polyline_data(curve_file), semilandmarks)
            curves[subdir].append(curve)
        return curves
    
    def _load_all_curves(self, semilandmarks):
        curves = {}
        for subdir in self.subdirs:
            curves = self._load_curves_in_dir(subdir, curves, semilandmarks)
        return curves         
    
    def _load_io_error_curves(self, semilandmarks):
        subdir_abs = os.path.join(self.datafolder, self.io_error_subdir)
        curves = {}
        for curve_file in glob.glob(subdir_abs + '/*.asc'):
            logging.info(curve_file)
            m = re.search(".*\/(.*)\ (\d+)\..*$", curve_file)
            if m:      
                specimen_name = m.groups()[0]
                curve = subdivcurve.subdivide_curve(sampledata.load_polyline_data(curve_file), semilandmarks)
                if specimen_name not in curves:
                    curves[specimen_name] = []
                curves[specimen_name].append(curve)
        return curves   
    
    def analyze_variability(self, output_dir):
        self.riface.call_r('base/processcurves.R', ['--variability', "--output", output_dir])
    
    def analyze_io_error(self, output_dir):
        self.riface.call_r('base/processcurves.R', ['--io_error', "--output", output_dir])
    
    def _show_curves(self, data, filename=None, radius=0.001, res=(1024, 1024), opts=dict()):
        #
        if 'res' in opts:
            res = opts['res']
        if 'radius' in opts:
            radius = opts['radius']
        if 'filename' in opts:
            filename = opts['filename']

        #
        vdata = []
        for key, group in data.items():
            color = (1, 0, 0)
            if 'default_color' in opts:
                color = opts['default_color']
            if 'colors' in opts and key in opts['colors']:
                color = opts['colors'][key]
            for curve in group:
                if 'curve' in opts and opts['curve']:
                    sls = [dict(dat=sampledata.curve(curve, radius), col=color)]
                elif 'values' in opts:
                    sls = sampledata.create_balls(curve, radius, color=color, values=opts['values'])
                else:
                    sls = sampledata.create_balls(curve, radius, color=color)
                
                vdata = vdata + sls

        if 'gizmo' in opts:
            vdata = vdata + [dict(dat=sampledata.cube_gizmo_data(), col=(0, 1, 0))]

        # 
        v = viewer.Viewer(vdata, size=res)
        v.filename = filename
        if 'camera' in opts:
            v.set_camera(position=opts['camera']['position'], focal_point=opts['camera']['focal_point'], parallel_scale=opts['camera']['parallel_scale'])
        v.render()
    
    def visualize_all(self):
        data = self.riface.load_from_r("output/all_gpa.csv")
        groups = self.riface.load_csv("output/all_group.csv")
        for i in range(len(data[""])):
            tmpdata = {"": data[""][0:(i + 1)]}
            self._show_curves(tmpdata, "output/filename%04d_%s.png" % (i, groups[i][0]))

    def curve_diff(self, curve1, curve2):

        def diff_sq(val1, val2):
            return sum([(val1[i] - val2[i]) * (val1[i] - val2[i]) for i in range(3)])

        diff = []
        lms_count = len(curve1)
        for i in range(lms_count):
            val = diff_sq(curve1[i], curve2[i])
            diff.append(val)
        return diff
    
    def visualize_means(self, output_dir, prefix='means', opts=None):
        means = self.riface.load_from_r("output/" + prefix + ".csv")
        groups = self.riface.load_csv("output/" + prefix + "_group.csv")
        data = {}
        for mean_index in range(len(means[""])):
            data[groups[mean_index][0]] = [means[""][mean_index]]
        self._show_curves(data, opts=opts)

    def visualize_mean_difference(self, output_dir, prefix='means', opts=None):
        means = self.riface.load_from_r("output/" + prefix + ".csv")
        groups = self.riface.load_csv("output/" + prefix + "_group.csv")
        for diff_pair in opts['diffs']:
            indx1 = diff_pair[0]
            indx2 = diff_pair[1]
            opts['values'] = self.curve_diff(means[""][indx1], means[""][indx2])
            data = {}
            data[groups[indx1][0]] = [means[""][indx1]]
            self._show_curves(data, opts=opts)

    def preprocess_curves(self, semilandmarks, force=False):
        if self.riface.curve_files_uptodate() or force:
            curves = self._load_all_curves(semilandmarks)
            self.riface.store_for_r(curves)
        if self.riface.curve_files_uptodate('io_error') or force:
            self.riface.store_for_r(self._load_io_error_curves(semilandmarks), prefix='io_error')

