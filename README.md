# Machine Learning Fundamentals

A curated set of notebooks showing my hands-on work with core machine-learning concepts, from implementing regression algorithms with NumPy to using scikit-learn.

## Included notebooks

### `building_a_linear_regression_model.ipynb`
A self-contained multivariable linear-regression implementation using NumPy. It covers feature normalization, the cost function, analytical gradients, gradient descent, predictions, convergence plots, and error analysis.

### `linear_regression_with_scikit_learn.ipynb`
A regression notebook using `StandardScaler` and `SGDRegressor`, with prediction checks and feature-by-feature visualizations.

**Environment note:** this notebook was developed in Google Colab and uses helper utilities and `houses.txt` from the original learning environment (`lab_utils_multi.py`, `lab_utils_common.py`, and `deeplearning.mplstyle`). Those external helper files are not included in this repository, so this notebook is kept here primarily to show the scikit-learn workflow and code I wrote around it.

### `multivariable_linear_regression.ipynb`
A smaller from-scratch multivariable regression experiment implementing normalization, cost, gradients, gradient descent, predictions, and visualizations.

### `logistic_regression_from_scratch.ipynb`
Binary logistic regression implemented with NumPy, including sigmoid, log-loss, analytical gradients, gradient descent, prediction, accuracy calculation, and visualization.

## Why these notebooks are grouped together

These are focused learning implementations and experiments rather than full standalone applied projects. Grouping them keeps the GitHub profile clean while still showing the underlying ML fundamentals and progression from manual implementations to library-based workflows.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook
```

The three from-scratch notebooks are self-contained. The scikit-learn notebook still relies on the external learning-environment helper files described above.
