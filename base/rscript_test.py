#!/usr/bin/python3

""" Test calling r interpreter from python."""

import sys
from base import rscriptsupport

if __name__ == "__main__":
    rscriptsupport.RScripInterface(sys.argv[1]).call_r("rscript_test.R")
