#!/usr/bin/python3

""" Generate report with estimate analysis. """

import datetime
import logging
import pickle
import os

from weasyprint import HTML, CSS
from jinja2 import Template

from analyze import loadData

def generatePdf(html, filename):
    html = HTML(string=html, base_url='../output')
    html.write_pdf(filename,
        stylesheets=[CSS(string='img { width: 100px; }')])

def generateHtml(template, data):
    template = Template(template)
    return template.render(**data)

def getTemplate(filename):
    return open(filename).read()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    template = getTemplate("projects/aurikular/report.jinja2")

    dataframe = loadData("../output/sample_estimates.csv")
    results = pickle.load(open("../output/analysis_result.pickle", 'rb'))

    data = {
        "today": datetime.datetime.now(),
        "project_name": os.path.dirname(__file__)
    }

    html = generateHtml(template, data)

    generatePdf(html, '../output/report.pdf')
