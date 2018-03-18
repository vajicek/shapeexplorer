""" Support for calling r interpreter from python."""

import csv
import itertools
import logging
import os
import subprocess

class RScripInterface(object):

    def __init__(self, output):
        self.output = output

    def call_r(self, script, args=[]):
        cmd = ['Rscript', script] + args
        process = subprocess.Popen(' '.join(cmd), shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            print(line.decode('utf-8'), end='')
    
    
    def write_single_curve(self, category, curve_list):
        logging.info("processing category: " + category)
        with open(os.path.join(self.output, category + '.csv'), 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for curve in curve_list:
                curve_line = list(itertools.chain.from_iterable(curve))
                spamwriter.writerow([str(num) for num in curve_line])
    
    
    def curve_files_uptodate(self, prefix='all'):
        if os.path.isfile(os.path.join(self.output, prefix + '_group.csv')) or os.path.isfile(os.path.join(self.output, prefix + '.csv')):
            return False
        return True   
    
    
    def store_for_r(self, curves, prefix='all'):
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
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            landmarks = []
            for row in spamreader:
                landmarks.append(row)
            return landmarks
    
    
    def load_from_r(self, filename, dim=3):
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
