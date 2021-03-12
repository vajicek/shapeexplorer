#!/usr/bin/python3

""" Analyze auricular shape. """

import logging
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

import scipy.stats as stat

import statsmodels.api as sm

from common import OUTPUT, DESCRIPTORS, ESTIMATES, ANALYSIS

def _predictionColName(indep, dep, type='loo'):
    return type + '_' + ("_".join(dep)) + "_by_" + ('_'.join(indep))

def _oneLeaveOutPredictions(dataframe, model, indep, dep):
    loo = LeaveOneOut()
    prediction_column = _predictionColName(indep, dep)
    dataframe[prediction_column] = 0
    for train, test in loo.split(dataframe):
        train_indices = dataframe.index[train]
        test_indices = dataframe.index[test]
        X = dataframe.loc[train_indices][indep]
        y = dataframe.loc[train_indices][dep]

        fit = model.fit(X, y.values.ravel())
        prediction = fit.predict(dataframe.loc[test_indices][indep])

        dataframe.at[test_indices, prediction_column] = np.exp(prediction)

def _modelStatistics(X, y, indep):
    X_with_constants = sm.add_constant(X)
    model = sm.OLS(y, X_with_constants)
    results = model.fit()
    return dict(pvalue=results.pvalues[indep].values[0])

def _getPvalue(fit, X, y):
    sse = np.sum((fit.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
    se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
    t = fit.coef_ / se
    p = (2 * (1 - stat.t.cdf(np.abs(t), y.shape[0] - X.shape[1])))
    return p[0]

def _modelPredictions(dataframe, model, indep, dep):
    prediction_column = _predictionColName(indep, dep, 'train')
    X = dataframe[indep]
    y = dataframe[dep]
    fit = model.fit(X, y.values.ravel())

    prediction = fit.predict(dataframe[indep])
    dataframe[prediction_column] = np.exp(prediction)

    #return _modelStatistics(X, y, indep)
    return dict(pvalue=_getPvalue(fit, X, y.values.ravel()))

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


def _evaluateModel(dataframe, indep, model=LinearRegression(), dep=['logAge']):
    model_stats = _modelPredictions(dataframe, model, indep, dep)
    _oneLeaveOutPredictions(dataframe, model, indep=indep, dep=dep)

    age_predictions = dataframe[_predictionColName(indep, dep)]
    age = dataframe['age']
    model_stats.update(_computeStatistics(age, age_predictions))

    return model_stats

def _evaluateAllModels(dataframe):
    indeps = [['logSAH'], ['logBE'], ['VC'], ['logSAH', 'VC'], ['logBE', 'VC']]
    subsets = ['all']
    results = []
    for subset in subsets:
        for indep in indeps:
            print('Evaluating: logAge ~ %s' % indep)
            model_stats = _evaluateModel(dataframe, indep)
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
    plots=[]
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
    model_results = _evaluateAllModels(dataframe)
    analysis_result = dict(
        model_results=model_results,
        age_descriptor=age_descriptor,
        age_histogram=age_histogram)

    _saveData(dataframe, os.path.join(folder, ESTIMATES))
    pickle.dump(analysis_result, open(os.path.join(folder, ANALYSIS), 'wb'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    analyze(OUTPUT)
