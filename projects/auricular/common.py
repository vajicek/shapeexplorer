""" Common constants for the project. """

import glob
import os
import re

OUTPUT = "../output"
DATAFOLDER = os.path.expanduser("~/data/aurikularni_plocha_ply2/")

SAMPLE = 'sample.csv'
DESCRIPTORS = 'sample_descriptors.csv'
ESTIMATES = 'sample_estimates.csv'
ANALYSIS = 'analysis_result.pickle'

REPORT_TEMPLATE = "report.jinja2"
LIST_TEMPLATE = "list.jinja2"
FFT_REPORT_TEMPLATE = "fft_report.jinja2"
CURVATURE_REPORT_TEMPLATE = "curvature_report.jinja2"

FILENAME_PATTERN = re.compile(
    r'.*/(.*)(S|Cr|Th|Co1|Co2)_(aur)_(dex|sin)_(F|M)([0-9]*)')

def _parse_name(filename):
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return {'basename': os.path.basename(filename),
            'filename': filename,
            'name': match.group(1),
            'subset': match.group(2),
            'type': match.group(3),
            'side': match.group(4),
            'sex': match.group(5),
            'age': match.group(6)
            }

def get_sample(input_folder):
    ply_files_glob = os.path.join(os.path.expanduser(input_folder), "*.ply")
    for abs_filename in glob.glob(ply_files_glob):
        yield _parse_name(abs_filename)
