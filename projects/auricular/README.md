# Auricular surface - age prediction

## Dependencies

* jupyter-notebook, scikit-learn, tensorflow, pandas, numpy, mathplotlib, trimesh

## Descriptors

### DNE - Dirichlet normal energy - curvature-like local surface metric

* Orientation, position invariant
* Fast implementation: https://sshanshans.github.io/articles/ariadne.html
* Parameters: sampling, neighbourhood size

### Distance map transform

* In principle orientation, position invariant
* Computed from projection to 2D
* Holes filling

## Statistics of DNE

* Mean DNE
* Histogram of DNE

## Combined distance and DNE

* 2D histogram of DNE x distance

## Extract descriptors

* Implementation in curvaturedescriptors.py
* Exection via jupyter notebook curvaturedescriptors.ipynb

## Jupyter notebook

* Run in auricular folder

```
jupyter-notebook
```


## Machine learning

* Age predciction

### "Traditional" methods

* LinearRegression, SVM (scikit-learn)
* ANN - multilayer perceptron (tensorflow)

### Neural networks

* Execute via curvaturedescriptors_age_prediction.ipynb
* Experiment on traning/testing/validation
* Evaluate model with 10-fold cross-validation

* Monitor progress

```
tensorboard --logdir=./mylogs --port=6006
```

## Dataset

* 812 specimens, age (mean: 54.36, std: 18.96)

## Result

* Expressed via RMSE
* Linear regression on mean DNE
* SVM regresion on mean DNE
* SVM regresion on histogram of DNE
* ANN on 2D histrogram