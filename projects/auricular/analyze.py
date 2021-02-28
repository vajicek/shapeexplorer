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

import statsmodels.api as sm

def predictionColName(indep, dep, type='loo'):
    return type + '_' + ("_".join(dep)) + "_by_" + ('_'.join(indep))

def oneLeaveOutPredictions(dataframe, model, indep, dep):
    loo = LeaveOneOut()
    prediction_column = predictionColName(indep, dep)
    dataframe[prediction_column] = 0
    for train, test in loo.split(dataframe):
        train_indices = dataframe.index[train]
        test_indices = dataframe.index[test]
        X = dataframe.loc[train_indices][indep]
        y = dataframe.loc[train_indices][dep]

        fit = model.fit(X, y.values.ravel())
        prediction = fit.predict(dataframe.loc[test_indices][indep])

        dataframe.at[test_indices, prediction_column] = np.exp(prediction)

def modelStatistics(X, y, indep):
    X_with_constants = sm.add_constant(X)
    model = sm.OLS(y, X_with_constants)
    results = model.fit()
    return dict(pvalue=results.pvalues[indep].values[0])

def modelPredictions(dataframe, model, indep, dep):
    prediction_column = predictionColName(indep, dep, 'train')
    X = dataframe[indep]
    y = dataframe[dep]
    fit = model.fit(X, y.values.ravel())

    prediction = fit.predict(dataframe[indep])
    dataframe[prediction_column] = np.exp(prediction)

    return modelStatistics(X, y, indep)


def loadData(filename):
    return pd.read_csv(filename, sep=',', quotechar='"')

def saveData(dataframe, filename):
    return dataframe.to_csv(filename, sep=',', quotechar='"')

def addLogColumns(data, columns):
    for column_pair in columns:
        data[column_pair[1]] = np.log(data[column_pair[0]])

def computeStatistics(actual, preditions):
    mse = mean_squared_error(actual, preditions)
    rmse = np.sqrt(mse)
    bias = np.sum(actual - preditions) / preditions.size
    inaccuracy = np.sum(np.abs(actual - preditions)) / preditions.size
    return dict(rmse=rmse, bias=bias, inaccuracy=inaccuracy)


def evaluateModel(dataframe, indep, model=LinearRegression(), dep=['logAge']):
    model_stats = modelPredictions(dataframe, model, indep, dep)
    oneLeaveOutPredictions(dataframe, model, indep=indep, dep=dep)

    age_predictions = dataframe[predictionColName(indep, dep)]
    age = dataframe['age']
    model_stats.update(computeStatistics(age, age_predictions))

    return model_stats

def evaluateAllModels(dataframe):
    indeps = [['logSAH'], ['logBE'], ['VC'], ['logSAH', 'VC'], ['logBE', 'VC']]
    subsets = ['all']
    results = []
    for subset in subsets:
        for indep in indeps:
            print('Evaluating: logAge ~ %s' % indep)
            model_stats = evaluateModel(dataframe, indep)
            model_stats.update(dict(indep=indep, subset=subset))
            results.append(model_stats)
    return results

def genAgeDescriptorPlots(dataframe):
    plots=[]
    for x in ['BE', 'SAH', 'VC']:
        filename = 'scatter_' + x + '.png'
        output_filepath = os.path.join('../output/', filename)
        plots.append({'filename': filename})
        dataframe.plot.scatter(x=x, y='age')
        plt.savefig(output_filepath, dpi=100)
    return plots

def genAgeHistogram(dataframe):
    plt.figure(figsize=(10, 3))
    dataframe['age'].hist()
    filename = 'age_histogram.png'
    output_filepath = os.path.join('../output/', filename)
    plt.savefig(output_filepath, dpi=100)
    return {'filename': filename}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataframe = loadData('../output/sample_descriptors.csv')

    addLogColumns(dataframe, [('age', 'logAge'), ('BE', 'logBE'), ('SAH', 'logSAH')])

    age_histogram = genAgeHistogram(dataframe)
    age_descriptor = genAgeDescriptorPlots(dataframe)
    model_results = evaluateAllModels(dataframe)
    analysis_result = dict(
        model_results=model_results,
        age_descriptor=age_descriptor,
        age_histogram=age_histogram)

    saveData(dataframe, "../output/sample_estimates.csv")
    pickle.dump(analysis_result, open("../output/analysis_result.pickle", 'wb'))
