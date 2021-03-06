
OUTPUT = "../output"
DATAFOLDER = "~/data/aurikularni_plocha_ply/"

SAMPLE = 'sample.csv'
DESCRIPTORS = 'sample_descriptors.csv'
ESTIMATES = 'sample_estimates.csv'
ANALYSIS = 'analysis_result.pickle'

REPORT_TEMPLATE = "report.jinja2"
LIST_TEMPLATE = "list.jinja2"


def getTemplateFile(filename):
    return os.path.join(os.path.dirname(__file__), filename)
