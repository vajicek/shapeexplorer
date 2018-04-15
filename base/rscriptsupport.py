""" Support for calling r interpreter from python."""

import csv
import itertools
import logging
import os
import subprocess


class RScripInterface(object):
    """ Interface class for R: call Rscript, read and write input/output data."""

    def __init__(self, output):
        self.output = output

    def call_r(self, script, args=[]):
        """ Call Rscript and pass output to print."""
        cmd = ['Rscript', script] + args
        process = subprocess.Popen(' '.join(cmd), shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            print(line.decode('utf-8'), end='')
    
    def write_single_curve(self, category, curve_list):
        """ Write curves to .csv file curve-by-line."""
        logging.info("processing category: " + category)
        with open(os.path.join(self.output, category + '.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for curve in curve_list:
                curve_line = list(itertools.chain.from_iterable(curve))
                spamwriter.writerow([str(num) for num in curve_line])
    
    def curve_files_uptodate(self, prefix='all'):
        """ Return true if prefix.csv and prefix_group.csv exist."""
        if os.path.isfile(os.path.join(self.output, prefix + '_group.csv')) or os.path.isfile(os.path.join(self.output, prefix + '.csv')):
            return False
        return True   
    
    def store_for_r(self, curves, prefix='all'):
        """ Store dictionary of curve lists: curve per file and all curves to
        a single file with prefix + prefix_group.csv (for preserving groups)."""
        # store separate lists
        for category, curve_list in curves.items():
            self.write_single_curve(category, curve_list)
    
        # store all curves
        all_group_filename = os.path.join(self.output, prefix + '_group.csv')
        all_filename = os.path.join(self.output, prefix + '.csv')
        with open(all_group_filename, 'w') as all_group_csvfile, open(all_filename, 'w') as all_csvfile:
            all_groupwriter = csv.writer(all_group_csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            allwriter = csv.writer(all_csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for category, curve_list in curves.items():
                for curve in curve_list:
                    curve_line = list(itertools.chain.from_iterable(curve))
                    allwriter.writerow([str(num) for num in curve_line])
                    all_groupwriter.writerow([category])
    
    def load_csv(self, filename):
        """ Read data from CSV."""
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks = []
            for row in spamreader:
                landmarks.append(row)
            return landmarks
    
    def write_csv(self, filename, groups):
        """ Write data to CSV."""
        filename_path = os.path.join(self.output, filename + '.csv')
        ofile = open(filename_path, "w")
        writer = csv.writer(ofile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        order = 1
        for group_name, group in groups.items():
            for row in group:
                writer.writerow([order, group_name] + (row if isinstance(row, list) else [row]))
                order = order + 1
        ofile.close()
    
    def load_from_r(self, filename, dim=3):
        """ Load landmarks from bigtable file to list of landmarks."""
        retval = {}
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            landmarks = []
            for row in spamreader:
                landmarks_count = int(len(row) / dim)
                row = [float(r) for r in row]
                lms = [row[i * dim:(i + 1) * dim] for i in range(landmarks_count)]
                landmarks.append(lms)
            retval[""] = landmarks 
        return retval
