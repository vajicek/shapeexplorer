#!/usr/bin/python3

""" Generate report with estimate analysis. """

import datetime
import logging
import pickle
import os

from weasyprint import HTML, CSS
from jinja2 import Template

from analyze import loadData

from common import OUTPUT, SAMPLE, ESTIMATES, ANALYSIS
from common import REPORT_TEMPLATE, LIST_TEMPLATE

def _generatePdf(html, filename, base_url):
    html = HTML(string=html, base_url=base_url)
    html.write_pdf(filename)

def _generateHtml(template, data):
    template = Template(template)
    return template.render(**data)

def _getTemplate(filename):
    return open(filename).read()

def _getTemplateFile(filename):
    return os.path.join(os.path.dirname(__file__), filename)

class Report:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def _getOutputFile(self, filename):
        return os.path.join(self.output_dir, filename)

    def generateReport(self):
        template = _getTemplate(_getTemplateFile(REPORT_TEMPLATE))

        dataframe = loadData(self._getOutputFile(ESTIMATES))

        analysis_result = pickle.load(open(self._getOutputFile(ANALYSIS), 'rb'))

        now = datetime.datetime.now()
        data = {
            "today": now,
            "project_name": os.path.basename(os.path.dirname(__file__)),
            "dataframe": dataframe,
            "sample_cols": ("name", "sex", "age", "side"),
            "descriptors_cols": ("name", "age", "BE", "SAH", "VC"),
            "predicted_cols": ("name", "age", "loo_logAge_by_logBE", "loo_logAge_by_logSAH", "loo_logAge_by_VC"),
            "shortened": True,
            "analysis_result": analysis_result,
            "describe": {
                "side": dataframe['side'].value_counts(),
                "sex": dataframe['sex'].value_counts(),
                "age": dataframe['age'].describe(),
                "subset": dataframe['subset'].value_counts(),
            }
        }

        html = _generateHtml(template, data)

        _generatePdf(html, self._getOutputFile('report_%s.pdf' % now.strftime("%Y%m%d")), self.output_dir)

    def generateList(self):
        template = _getTemplate(_getTemplateFile(LIST_TEMPLATE))

        dataframe = loadData(self._getOutputFile(ESTIMATES))

        images = []
        for i in dataframe.index:
            imagefile = os.path.splitext(dataframe['basename'][i])[0] + ".png"
            images.append(imagefile)
        dataframe['imagefile'] = images

        dataframe = dataframe.sort_values('age')

        now = datetime.datetime.now()
        data = {
            "today": now,
            "project_name": os.path.dirname(__file__),
            "dataframe": dataframe,
        }

        html = _generateHtml(template, data)
        open(self._getOutputFile('list.html'), 'w').write(html)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    report = Report(OUTPUT)
    report.generateReport()
    report.generateList()
