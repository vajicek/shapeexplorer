#!/usr/bin/python3

""" Generate report with estimate analysis. """

import datetime
import logging
import pickle
import os

from weasyprint import HTML, CSS
from jinja2 import Template

from analyze import loadData

import common

def _generatePdf(html, filename, base_url, pdf_css):
    html = HTML(string=html, base_url=base_url)
    html.write_pdf(filename, stylesheets=[CSS(string=pdf_css)], optimize_images=True)

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
        template = _getTemplate(_getTemplateFile(common.REPORT_TEMPLATE))

        dataframe = loadData(self._getOutputFile(common.ESTIMATES))

        analysis_result = pickle.load(open(self._getOutputFile(common.ANALYSIS), 'rb'))

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
        template = _getTemplate(_getTemplateFile(common.LIST_TEMPLATE))

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
            "project_name": os.path.basename(os.path.dirname(__file__)),
            "dataframe": dataframe,
        }

        html = _generateHtml(template, data)
        open(self._getOutputFile('list.html'), 'w').write(html)

    def generateFft(self):
        template = _getTemplate(_getTemplateFile(common.FFT_REPORT_TEMPLATE))

        now = datetime.datetime.now()
        data = {
            "today": now,
            "project_name": os.path.basename(os.path.dirname(__file__)),
        }

        html = _generateHtml(template, data)

        _generatePdf(html, self._getOutputFile('fft_report_%s.pdf' % now.strftime("%Y%m%d")), self.output_dir)

    def generateCurvature(self, data_dict, pdf_css):
        template = _getTemplate(_getTemplateFile(common.CURVATURE_REPORT_TEMPLATE))

        now = datetime.datetime.now()
        date_str = now.strftime("%Y%m%d")
        data = {
            "today": now,
            "date_str": date_str,
            "project_name": os.path.basename(os.path.dirname(__file__)),
            **data_dict
        }

        html = _generateHtml(template, data)

        open(self._getOutputFile('curvature.html'), 'w').write(html)
        _generatePdf(html, self._getOutputFile('curvature_%s.pdf' % date_str), self.output_dir, pdf_css)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    report = Report(common.OUTPUT)
    report.generateReport()
    report.generateList()
