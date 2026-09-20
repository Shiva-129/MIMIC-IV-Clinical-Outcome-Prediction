
# Model Hyperparameter Tuning

This document describes the hyperparameter search spaces and selection strategies
used for all predictive models.

## Logistic Regression
- Search: Grid search on validation set
- Selection metric: Macro-averaged ROC-AUC

```text
C: [0.001, 0.01, 0.1, 1.0, 10.0]
penalty: l2
solver: lbfgs, saga
````

## Random Forest

* Search: Random search (20 sampled combinations)
* Selection metric: ROC-AUC

```text
n_estimators: [100, 200, 300]
max_depth: [10, 15, 20, None]
min_samples_split: [5, 10, 20]
min_samples_leaf: [2, 5, 10]
```

## XGBoost

* Separate model trained per outcome
* Search: Random search (15 combinations per outcome)
* Selection metric: ROC-AUC

```text
n_estimators: [100, 200, 300]
max_depth: [4, 6, 8]
learning_rate: [0.01, 0.05, 0.1]
subsample: [0.7, 0.8, 0.9]
colsample_bytree: [0.7, 0.8, 0.9]
```

## Multi-Layer Perceptron

* Search: Grid search
* Early stopping enabled (validation_fraction = 0.1)

```text
hidden_layer_sizes: [(128,64), (128,64,32), (256,128,64)]
learning_rate_init: [0.001, 0.01]
alpha: [0.0001, 0.001, 0.01]
activation: relu
solver: adam
max_iter: 300
```