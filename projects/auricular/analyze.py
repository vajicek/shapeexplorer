#!/usr/bin/python3

""" Analyze auricular shape. """

import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

def predictionColName(indep, dep):
    return ("_".join(dep)) + "_by_" + ('_'.join(indep))

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

def loadData(filename):
    return pd.read_csv(filename, sep=',', quotechar='"')

def saveData(dataframe, filename):
    return dataframe.to_csv(filename, sep=',', quotechar='"')

def addLogColumns(data, columns):
    for column_pair in columns:
        data[column_pair[1]] = np.log(data[column_pair[0]])

def evaluateModel(dataframe, indep, model=LinearRegression(), dep=['logAge']):
    oneLeaveOutPredictions(dataframe, model, indep=indep, dep=dep)
    age_predictions = dataframe[predictionColName(indep, dep)]
    age = dataframe['age']
    mse = mean_squared_error(age, age_predictions)
    rmse = np.sqrt(mse)
    return rmse

def evaluateAllModels(dataframe):
    indeps = [['logSAH'], ['logBE'], ['VC'], ['logSAH', 'VC'], ['logBE', 'VC']]
    results = []
    for indep in indeps:
        print('Evaluating: logAge ~ %s' % indep)
        rmse = evaluateModel(dataframe, indep)
        results.append(dict(indep=indep, rmse=rmse))
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataframe = loadData('../output/sample_descriptors.csv')
    addLogColumns(dataframe, [('age', 'logAge'), ('BE', 'logBE'), ('SAH', 'logSAH')])

    results = evaluateAllModels(dataframe)

    saveData(dataframe, "../output/sample_estimates.csv")
    pickle.dump(results, open("../output/analysis_result.pickle", 'wb'))
