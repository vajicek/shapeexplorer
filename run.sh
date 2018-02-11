#!/bin/bash

# on ubuntu install vtk for python: pip3 install vtk

export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/vtk/:${LD_LIBRARY_PATH}
#python3 viewer_test.py
#python3 rscript_test.py
python3 processcurves.py
