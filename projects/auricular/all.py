#!/usr/bin/python3

from common import OUTPUT, DATAFOLDER

from preprocessing import preprocessing
from analyze import analyze
from report import Report

if __name__ == "__main__":
    preprocessing(DATAFOLDER, OUTPUT)
    analyze(OUTPUT)
    report = Report(OUTPUT)
    report.generateReport()
    report.generateList()
