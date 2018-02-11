""" Support for calling r interpreter from python."""

import csv
import itertools
import logging
import os
import subprocess

OUTPUTFOLDER = "output"


def call_r(script):
    cmd = ['Rscript', script]
    process = subprocess.Popen(' '.join(cmd), shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode('utf-8'), end='')


def write_single_curve(category, curve_list):
    logging.info("processing category: " + category)
    with open(os.path.join(OUTPUTFOLDER, category + '.csv'), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for curve in curve_list:
            curve_line = list(itertools.chain.from_iterable(curve))
            spamwriter.writerow([str(num) for num in curve_line])


def curve_files_uptodate():
    if os.path.isfile(os.path.join(OUTPUTFOLDER, 'all_group.csv')) or os.path.isfile(os.path.join(OUTPUTFOLDER, 'all.csv')):
        return False
    return True   


def store_for_r(curves):
    # store separate lists
    for category, curve_list in curves.items():
        write_single_curve(category, curve_list)

    # store all curves
    all_group_filename = os.path.join(OUTPUTFOLDER, 'all_group.csv')
    all_filename = os.path.join(OUTPUTFOLDER, 'all.csv')
    with open(all_group_filename, 'w') as all_group_csvfile, open(all_filename, 'w') as all_csvfile:
        all_groupwriter = csv.writer(all_group_csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        allwriter = csv.writer(all_csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for category, curve_list in curves.items():
            for curve in curve_list:
                curve_line = list(itertools.chain.from_iterable(curve))
                allwriter.writerow([str(num) for num in curve_line])
                all_groupwriter.writerow([category])


def load_csv(filename):
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        landmarks = []
        for row in spamreader:
            landmarks.append(row)
        return landmarks


def load_from_r(filename, dim=3):
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
