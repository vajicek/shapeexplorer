#!/usr/bin/python3

from projects.tibiacurve import common  

IO_ERROR_OUTPUT_DIR='/home/vajicek/Dropbox/TIBIA/CURVATURE/results/io_error'
IO_ERROR_OUTPUT_LOG_BY_SLM='output_sm%02d.txt'

def analyze_io_error_slm(slm, output_dir, log_file):
    curves_processor = common.get_processor(output_dir, log_file)
    curves_processor.preprocess_curves(slm, True)
    curves_processor.analyze_io_error(output_dir)

for slm in [10, 20, 30]:
    analyze_io_error_slm(slm, IO_ERROR_OUTPUT_DIR, IO_ERROR_OUTPUT_LOG_BY_SLM % slm)