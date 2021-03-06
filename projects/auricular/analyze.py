""" Analyze auricular shape. """

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

import scipy.stats as stat

import statsmodels.api as sm

from .common import DESCRIPTORS, ESTIMATES, ANALYSIS

def _predictionColName(indep, dep, prefix='loo'):
    return prefix + '_' + ("_".join([str(d) for d in dep])) + "_by_" + ('_'.join([str(i) for i in indep]))


def removeOutliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def _oneLeaveOutPredictions(dataframe, model, indep, dep):
    loo = LeaveOneOut()
    prediction_column = _predictionColName(indep, dep)
    dataframe[prediction_column] = 0
    for train, test in loo.split(dataframe):
        train_indices = dataframe.index[train]
        test_indices = dataframe.index[test]
        x = dataframe.loc[train_indices][indep]
        y = dataframe.loc[train_indices][dep]

        fit = model.fit(x, y.values.ravel())
        prediction = fit.predict(dataframe.loc[test_indices][indep])

        dataframe.at[test_indices, prediction_column] = np.exp(prediction)


def _modelStatistics(x, y, indep):
    x_with_constants = sm.add_constant(x)
    model = sm.OLS(y, x_with_constants)
    results = model.fit()
    return dict(pvalue=results.pvalues[indep].values[0])


def _getPvalue(fit, x, y):
    sum_sq_err = np.sum((fit.predict(x) - y) ** 2, axis=0) / float(x.shape[0] - x.shape[1])
    sq_err = np.array([np.sqrt(np.diagonal(sum_sq_err * np.linalg.inv(np.dot(x.T, x))))])
    t_value = fit.coef_ / sq_err
    p_value = 2 * (1 - stat.t.cdf(np.abs(t_value), y.shape[0] - x.shape[1]))
    return p_value[0]


def _modelPredictions(dataframe, model, indep, dep):
    prediction_column = _predictionColName(indep, dep, 'train')
    x = dataframe[indep]
    y = dataframe[dep]
    fit = model.fit(x, y.values.ravel())

    prediction = fit.predict(dataframe[indep])
    dataframe.loc[:, prediction_column] = np.exp(prediction)

    # return _modelStatistics(X, y, indep)
    # return dict(pvalue=_getPvalue(fit, X, y.values.ravel()))
    return dict(pvalue=0)


def _saveData(dataframe, filename):
    return dataframe.to_csv(filename, sep=',', quotechar='"')


def _addLogColumns(data, columns):
    for column_pair in columns:
        data[column_pair[1]] = np.log(data[column_pair[0]])


def _computeStatistics(actual, preditions):
    mse = mean_squared_error(actual, preditions)
    rmse = np.sqrt(mse)
    bias = np.sum(actual - preditions) / preditions.size
    inaccuracy = np.sum(np.abs(actual - preditions)) / preditions.size
    return dict(rmse=rmse, bias=bias, inaccuracy=inaccuracy)


def _evaluateModel(dataframe, indep, model, dep=None):
    dep = dep or ['logAge']

    model_stats = _modelPredictions(dataframe, model, indep, dep)
    _oneLeaveOutPredictions(dataframe, model, indep=indep, dep=dep)

    age_predictions = dataframe[_predictionColName(indep, dep)]
    age = dataframe['age']
    model_stats.update(_computeStatistics(age, age_predictions))

    return model_stats


def evaluateAllModels(dataframe, indeps=None, subsets=None, dep=None, model=LinearRegression()):
    indeps = indeps or [['logSAH'], ['logBE'], ['VC'], ['logSAH', 'VC'], ['logBE', 'VC']]
    subsets = subsets or ['all']
    dep = dep or ['logAge']

    results = []
    for subset in subsets:
        for indep in indeps:
            print('Evaluating: %s ~ %s' % (dep, indep))
            model_stats = _evaluateModel(dataframe, indep, model, dep=dep)
            model_stats.update(dict(indep=indep, subset=subset))
            results.append(model_stats)

        # add benchmark - mean age
        age = dataframe['age']
        model_stats = dict(pvalue=0)
        model_stats.update(_computeStatistics(age, np.array([age.mean()] * len(age))))
        model_stats.update(dict(indep=['mean'], subset=subset))
        results.append(model_stats)

        # add benchmark - random age
        model_stats = dict(pvalue=0)
        random_age = np.random.normal(loc=age.mean(), scale=age.std(), size=len(age))
        model_stats.update(_computeStatistics(age, random_age))
        model_stats.update(dict(indep=['random'], subset=subset))
        results.append(model_stats)

    return results


def _genAgeDescriptorPlots(folder, dataframe):
    plots = []
    for x in ['BE', 'SAH', 'VC', 'logBE', 'logSAH']:
        filename = 'scatter_' + x + '.png'
        output_filepath = os.path.join(folder, filename)
        plots.append({'filename': filename})
        dataframe.plot.scatter(x='age', y=x)
        plt.savefig(output_filepath, dpi=100)
    return plots


def _genAgeHistogram(folder, dataframe):
    plt.figure(figsize=(10, 3))
    dataframe['age'].hist()
    filename = 'age_histogram.png'
    output_filepath = os.path.join(folder, filename)
    plt.savefig(output_filepath, dpi=100)
    return {'filename': filename}


def loadData(filename):
    return pd.read_csv(filename, sep=',', quotechar='"')


def analyze(folder):
    dataframe = loadData(os.path.join(folder, DESCRIPTORS))

    _addLogColumns(dataframe, [('age', 'logAge'), ('BE', 'logBE'), ('SAH', 'logSAH')])

    age_histogram = _genAgeHistogram(folder, dataframe)
    age_descriptor = _genAgeDescriptorPlots(folder, dataframe)
    model_results = evaluateAllModels(dataframe)
    analysis_result = dict(
        model_results=model_results,
        age_descriptor=age_descriptor,
        age_histogram=age_histogram)

    _saveData(dataframe, os.path.join(folder, ESTIMATES))
    pickle.dump(analysis_result, open(os.path.join(folder, ANALYSIS), 'wb'))


class ModelAnalysis:

    def __init__(self, data, hist_descriptors, data_type='dist_curv'):
        self.data_type = data_type
        self.data = data
        self.hist_descriptors = hist_descriptors

    def modelForBins(self, bins, indeps=None, dist=2.0, model=LinearRegression(), normalize_dist=True):

        if self.data_type == 'dist_curv':
            dataframe = pd.DataFrame(self.hist_descriptors[dist].getSampleHistogram2dData(bins, True, normalize_dist))
        elif self.data_type == 'curv':
            dataframe = pd.DataFrame(self.hist_descriptors[dist].getSampleHistogramData(bins))
        else:
            print("Unknown data_type %s" % self.data_type)

        dataframe['age'] = [float(data1['age']) for data1 in self.data]
        dataframe['logAge'] = np.log(dataframe['age'])
        indeps = indeps or [list(range(bins))]
        results = evaluateAllModels(dataframe, indeps=indeps, dep=['logAge'], model=model)
        return pd.DataFrame(results)

    def plotRmsePerBins(self, bins_rmse_list):
        bins, rmses = list(zip(*bins_rmse_list))
        dataframe = pd.DataFrame({
            'rmse': rmses,
            'bins': bins})
        dataframe.plot(y='rmse', x='bins')
        _ = plt.xticks(dataframe['bins'])

    def binsRmse(self, dist=1.0, model=LinearRegression()):
        for bins in range(2, 20):
            yield bins, self.modelForBins(bins, dist=dist, model=model)['rmse'][0]

    def compareMethods(self, dist=1.0):
        lsvr_bins_rmse_list = list(self.binsRmse(dist=dist, model=LinearSVR()))
        svr_bins_rmse_list = list(self.binsRmse(dist=dist, model=SVR()))
        lr_bins_rmse_list = list(self.binsRmse(dist=dist, model=LinearRegression()))

        dataframe = pd.DataFrame({
            'linear regression': list(zip(*lr_bins_rmse_list))[1],
            'linear SVR': list(zip(*lsvr_bins_rmse_list))[1],
            'SVR': list(zip(*svr_bins_rmse_list))[1],
            'bins': list(zip(*svr_bins_rmse_list))[0]})
        dataframe.plot.line(x='bins')
        _ = plt.xticks(dataframe['bins'])

    def twoParamPlot(self, dist=1.0, x_bin=0, y_bin=2, bins=3):
        x = [a[x_bin] for a in self.hist_descriptors[dist].getSampleHistogramData(bins)]
        y = [a[y_bin] for a in self.hist_descriptors[dist].getSampleHistogramData(bins)]
        age = [float(data1['age']) for data1 in self.data]
        pd.DataFrame({'x': x, 'y': y, 'age': age}).plot.scatter(x='x', y='y', c='age', colormap='viridis')
