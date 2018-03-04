#!/bin/bash

# on ubuntu install vtk for python: pip3 install vtk

export LC_TIME=C
export PYTHONPATH=${PYTHONPATH}:${PWD}
export LD_LIBRARY_PATH=/usr/local/lib/python3.6/dist-packages/vtk/:${LD_LIBRARY_PATH}
