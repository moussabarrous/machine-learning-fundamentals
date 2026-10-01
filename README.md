# Machine Learning Fundamentals

A curated set of notebooks showing my hands-on work with core machine-learning concepts, from implementing regression algorithms with NumPy to using scikit-learn and TensorFlow.

## Included notebooks

### `building_a_linear_regression_model.ipynb`
A self-contained multivariable linear-regression implementation using NumPy. It covers feature normalization, the cost function, analytical gradients, gradient descent, predictions, convergence plots, and error analysis.

### `linear_regression_with_scikit_learn.ipynb`
A regression notebook using `StandardScaler` and `SGDRegressor`, with prediction checks and feature-by-feature visualizations.

**Note:** This notebook depends on additional data and helper files.

### `multivariable_linear_regression.ipynb`
A smaller from-scratch multivariable regression experiment implementing normalization, cost, gradients, gradient descent, predictions, and visualizations.

### `logistic_regression_from_scratch.ipynb`
Binary logistic regression implemented with NumPy, including sigmoid, log-loss, analytical gradients, gradient descent, prediction, accuracy calculation, and visualization.

### `model_evaluation_and_selection.ipynb`
Model-selection experiments using train / cross-validation / test splits, feature scaling, polynomial regression, MSE comparison, neural-network model comparison, and classification error.

**Note:** This notebook depends on additional data and helper files.

## Why these notebooks are grouped together

These are focused learning implementations and experiments rather than full standalone applied projects. Grouping them keeps the GitHub profile clean while still showing the underlying ML fundamentals and progression from manual implementations to library-based workflows.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook
```

Some notebooks are self-contained; others depend on additional data and helper files.
