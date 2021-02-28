#!/usr/bin/python3

""" Generate report with estimate analysis. """

import datetime
import logging
import pickle
import os

from weasyprint import HTML, CSS
from jinja2 import Template

from analyze import loadData

def generatePdf(html, filename, base_url):
    html = HTML(string=html, base_url=base_url)
    html.write_pdf(filename)

def generateHtml(template, data):
    template = Template(template)
    return template.render(**data)

def getTemplate(filename):
    return open(filename).read()

class Report:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def getOutputFile(self, filename):
        return os.path.join(self.output_dir, filename)

    def generateReport(self):
        template_file = os.path.join(os.path.dirname(__file__), "report.jinja2")
        template = getTemplate(template_file)

        dataframe = loadData(self.getOutputFile('sample_estimates.csv'))

        analysis_result = pickle.load(open(self.getOutputFile('analysis_result.pickle'), 'rb'))

        now = datetime.datetime.now()
        data = {
            "today": now,
            "project_name": os.path.dirname(__file__),
            "dataframe": dataframe,
            "sample_cols": ("name", "sex", "age", "side"),
            "descriptors_cols": ("name", "age", "BE", "SAH", "VC"),
            "predicted_cols": ("name", "age", "loo_logAge_by_logBE", "loo_logAge_by_logSAH", "loo_logAge_by_VC"),
            "shortened": True,
            "analysis_result": analysis_result,
            "describe": {
                "side": dataframe['side'].describe(),
                "sex": dataframe['sex'].describe(),
                "age": dataframe['age'].describe(),
            }
        }

        html = generateHtml(template, data)

        #open(self.getOutputFile('index.html'),'w').write(html)

        generatePdf(html, self.getOutputFile('report_%s.pdf' % now.strftime("%Y%m%d")), self.output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    report = Report('../output')
    report.generateReport()
