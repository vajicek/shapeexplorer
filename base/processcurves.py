#!/usr/bin/python3

""" Process curves. """

import copy
import logging
import glob
import numpy as np
import os
import re

from base import sampledata
from base import subdivcurve
from base import rscriptsupport
from base import viewer


def _unslash(path):
    return path.replace("\\", "")


def _mkbasedir_if_not_exist(filename):
    if filename:
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)


class CurvesProcessor(object):
    """ Analyze groups of curves. """

    def __init__(self, datafolder, subdirs, io_error_subdir, output):
        """ datafolder - Data folder.
        subdirs - List of group subfolders.
        io_error_subdir - Folder with IO error data.
        output - Output folder. """
        self.datafolder = datafolder
        self.subdirs = subdirs
        self.io_error_subdir = io_error_subdir
        self.riface = rscriptsupport.RScripInterface(output)

    def get_groups_count(self):
        return len(self.subdirs)

    def length_analysis(self, output_dir):
        self.riface.call_r('base/processcurves.R',
                           ['--length_analysis', '--output', output_dir])

    def analyze_variability(self, input_dir, output_dir, slm_handling='none'):
        """ Analyze variability of the sample."""
        self.riface.call_r('base/processcurves.R',
                           ['--variability', '--input', input_dir, '--output',
                            output_dir, '--slm_handling', slm_handling])

    def analyze_allometry(self, input_dir, output_dir, slm_handling='none'):
        """ Analyze allometry of the sample groups."""
        self.riface.call_r('base/processcurves.R',
                           ['--allometry', '--input', input_dir, '--output',
                            output_dir, '--slm_handling', slm_handling])

    def analyze_io_error(self, input_dir, output_dir):
        """ Analyze io error via R."""
        self.riface.call_r('base/processcurves.R',
                           ['--io_error', '--input', input_dir,
                            '--output', output_dir])

    def visualize_all(self, input_dir, output_dir):
        """ Visualize all curves."""
        data = self.riface.load_from_r(os.path.join(input_dir, "all_gpa.csv"))
        groups = self.riface.load_csv(os.path.join(input_dir, "all_group.csv"))
        for i in range(len(data[""])):
            tmpdata = {"": data[""][0:(i + 1)]}
            self._show_curves(tmpdata,
                              os.path.join(output_dir,
                                           "filename%04d_%s.png" %
                                           (i, groups[i][0])))

    def visualize_means(self, input_dir, prefix='means', opts=None):
        """ Visualize means."""
        means = self.riface.load_from_r(
            os.path.join(input_dir, prefix + ".csv"))
        groups = self.riface.load_csv(
            os.path.join(input_dir, prefix + "_group.csv"))
        data = {}
        for mean_index in range(len(means[""])):
            data[groups[mean_index][0]] = [means[""][mean_index]]
        self._show_curves(data, opts=opts)

    def visualize_mean_difference(self, input_dir, prefix='means', opts=None):
        """ Visualize mean differences for given pairs."""
        means = self.riface.load_from_r(
            os.path.join(input_dir, prefix + ".csv"))
        groups = self.riface.load_csv(
            os.path.join(input_dir, prefix + "_group.csv"))
        for diff_pair in opts['diffs']:
            indx1 = diff_pair[0]
            indx2 = diff_pair[1]
            opts['values'] = self._curve_diff(means[""][indx1], means[""][indx2])
            opts['other'] = means[""][indx2]
            self._show_curves({groups[indx1][0]: [means[""][indx1]]}, opts=opts)

    def visualize_loadings(self, input_dir, output_dir, prefix='means',
                           opts=None):
        """ Visualize curve variability loadings."""
        means = self.riface.load_from_r(os.path.join(input_dir, prefix + ".csv"))
        groups = self.riface.load_csv(os.path.join(input_dir, prefix + "_group.csv"))
        loadings = self.riface.load_from_r(os.path.join(input_dir, "all_pca_loadings.csv"))
        opts['values'] = self._curve_dist(loadings[""][opts['pca_no']])
        opts['other'] = self._vectorize_loadings(means[""][7],
                                                 loadings[""][opts['pca_no']])
        self._show_curves({groups[7][0]: [means[""][7]]}, opts=opts)

    def preprocess_curves(self, semilandmarks, force=False):
        """ Preprocess curves and store to files for R."""
        if self.riface.curve_files_uptodate() or force:
            curves, names = self.load_all_curves(semilandmarks)
            self.riface.store_for_r(curves)
            self.riface.write_csv('names', names)
        if self.io_error_subdir and (self.riface.curve_files_uptodate('io_error') or force):
            self.riface.store_for_r(self._load_io_error_curves(semilandmarks), prefix='io_error')

    def load_all_curves(self, semilandmarks):
        """Load curves by groups and optionally """
        curves = {}
        names = {}
        for subdir in self.subdirs:
            curves = self._load_curves_in_dir(subdir, curves, names, semilandmarks)
        return curves, names

    def measure_length(self, curves):
        curve_lengths = {}
        for group_name, group_curves in curves.items():
            curve_lengths[group_name] = []
            for curve in group_curves:
                curve_len = subdivcurve.curve_length(curve)
                curve_lengths[group_name].append(curve_len)
        return curve_lengths

    def _vectorize_loadings(self, mean, loadings):
        loaded_mean = copy.deepcopy(mean)
        for lm, lm_loadings in zip(loaded_mean, loadings):
            lm[0] += lm_loadings[0]
            lm[1] += lm_loadings[1]
            lm[2] += lm_loadings[2]
        return loaded_mean

    def _load_curves_in_dir(self, subdir, curves, names, semilandmarks):
        subdir_abs = os.path.join(self.datafolder, subdir)
        curves[subdir] = []
        names[subdir] = []
        for curve_file in glob.glob(_unslash(subdir_abs) + '/**/*.asc', recursive=True):
            logging.info(curve_file)
            if semilandmarks:
                curve = subdivcurve.subdivide_curve(
                    sampledata.load_polyline_data(curve_file), semilandmarks)
            else:
                curve = sampledata.load_polyline_data(curve_file)
            curves[subdir].append(curve)
            names[subdir].append(curve_file)
        return curves

    def _load_io_error_curves(self, semilandmarks):
        subdir_abs = os.path.join(self.datafolder, self.io_error_subdir)
        curves = {}
        for curve_file in glob.glob(_unslash(subdir_abs) + '/*.asc'):
            logging.info(curve_file)
            m = re.search(r".*\/(.*)\ (\d+)\..*$", curve_file)
            if m:
                specimen_name = m.groups()[0]
                curve = subdivcurve.subdivide_curve(
                    sampledata.load_polyline_data(curve_file), semilandmarks)
                if specimen_name not in curves:
                    curves[specimen_name] = []
                curves[specimen_name].append(curve)
        return curves

    def _show_curves(self, data, filename=None, radius=0.001, res=(1024, 1024),
                     opts=dict()):
        if 'filename' in opts and type(opts['filename']) is list:
            camera_settings = opts['camera']
            filenames = opts['filename']
            for indx, filename in enumerate(filenames):
                opts['camera'] = camera_settings[indx]
                opts['filename'] = filename
                self._show_curves_view(data, filename=None, radius=radius,
                                       res=res, opts=opts)
        else:
            self._show_curves_view(data, filename=filename, radius=radius,
                                   res=res, opts=opts)

    def _create_curve_vdata(self, curve, opts, radius, normalization_factor,
                            color, arrows_scaling):
        vdata = []
        if 'normalize' in opts and opts['normalize']:
            if normalization_factor is None:
                normalization_factor = 1.0 / self._coord_var(curve)
            curve = self._normalize(curve, normalization_factor)
        if 'curve' in opts and opts['curve']:
            vdata += [dict(dat=sampledata.curve(curve, radius),
                           col=color)]
        elif 'values' in opts:
            if 'balls' not in opts or opts['balls']:
                vdata += sampledata.create_balls(curve, radius,
                                                 color=color,
                                                 values=opts['values'])
        else:
            vdata += sampledata.create_balls(curve, radius,
                                             color=color)

        if 'other' in opts:
            other_points = self._normalize(opts['other'],
                                           normalization_factor)
            vdata += sampledata.create_arrows(curve, arrows_scaling,
                                              color=color,
                                              other_points=other_points)
        return vdata

    def _show_vdata_in_viewer(self, vdata, filename, opts, res):
        v = viewer.Viewer(vdata, size=res)
        _mkbasedir_if_not_exist(filename)
        v.filename = filename
        if 'camera' in opts:
            v.set_camera(position=opts['camera']['position'],
                         focal_point=opts['camera']['focal_point'],
                         parallel_scale=opts['camera']['parallel_scale'],
                         view_up=opts['camera']['view_up'])
        v.render()

    def _show_curves_view(self, data, filename=None, radius=0.001,
                          res=(1024, 1024), opts=dict()):
        if 'res' in opts:
            res = opts['res']
        if 'radius' in opts:
            radius = opts['radius']
        if 'filename' in opts:
            filename = opts['filename']
        if 'arrows_scaling' in opts:
            arrows_scaling = opts['arrows_scaling']

        #
        vdata = []
        normalization_factor = None
        for key, group in data.items():
            color = (1, 0, 0)
            if 'default_color' in opts:
                color = opts['default_color']
            if 'colors' in opts and key in opts['colors']:
                color = opts['colors'][key]
            for curve in group:
                vdata += self._create_curve_vdata(curve, opts, radius,
                                                  normalization_factor, color,
                                                  arrows_scaling)

        if 'gizmo' in opts:
            vdata += [dict(dat=sampledata.cube_gizmo_data(), col=(0, 1, 0))]

        self._show_vdata_in_viewer(vdata, filename, opts, res)

    def _coord_var(self, curve):
        return np.std(np.array(curve))

    def _normalize(self, curve, scale):
        return [[v * scale for v in lm] for lm in curve]

    def _curve_dist(self, curve):

        def sq(val1):
            return sum([val1[i] * val1[i] for i in range(3)])

        dist_sq = []
        lms_count = len(curve)
        for i in range(lms_count):
            val = sq(curve[i])
            dist_sq.append(val)
        return dist_sq

    def _curve_diff(self, curve1, curve2):

        def diff_sq(val1, val2):
            return sum([(val1[i] - val2[i]) * (val1[i] - val2[i])
                        for i in range(3)])

        lms_diff_sq = []
        lms_count = len(curve1)
        for i in range(lms_count):
            val = diff_sq(curve1[i], curve2[i])
            lms_diff_sq.append(val)
        return lms_diff_sq
