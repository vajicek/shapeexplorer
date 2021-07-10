""" ANN ML utility functions, """

import numpy as np

from tensorflow import keras
from keras.layers import Dense, InputLayer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def buildModel(n_inputs=1, n_hidden_layers=2, n_neurons=1, learning_rate=0.0001):
    model = keras.Sequential()
    model.add(InputLayer(n_inputs))
    for _ in range(n_hidden_layers):
        model.add(Dense(n_neurons, kernel_initializer='he_uniform',  activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate,
                                                  beta_1=0.95,
                                                  beta_2=0.9995),
                  #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error')
    return model


def evaluateModel(x, y):
    model = buildModel(n_inputs=x.shape[1],
                       n_hidden_layers=2,
                       n_neurons=x.shape[1],
                       learning_rate=0.0001)

    k_fold = KFold(n_splits=10, shuffle=True)
    fold_rmse = []
    predicted = np.array([])
    predicted_indices = np.array([])
    for train_index, test_index in k_fold.split(x):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(x_train, y_train,
                  use_multiprocessing=True,
                  workers=8,
                  epochs=1000,
                  batch_size=10,
                  verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(patience=100)],
                  validation_data=(x_test, y_test))

        predictions = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(np.exp(predictions), np.exp(y_test)))
        fold_rmse += [rmse]
        print(rmse)
        predicted = np.append(predicted, predictions)
        predicted_indices = np.append(predicted_indices, test_index)

    return np.mean(fold_rmse), predicted, predicted_indices
