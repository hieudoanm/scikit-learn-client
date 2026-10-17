# skl — scikit-learn

Build a `Pipeline([("pre", ColumnTransformer), ("model", estimator)])`. Fit on train split only for supervised tasks. Classification metrics from `sklearn.metrics` with `average="macro"` and `zero_division=0`. Regression: RMSE, MAE, R². Do not leak test labels into the encoder (`handle_unknown="ignore"`). Pin sklearn in `pyproject.toml`.

## Task strategies

- **Classification / regression:** stratified (classification) or random hold-out
  using `splitRatio` and `randomSeed`; fit the preprocessor and estimator on the
  train split only.
- **Semi-supervised:** use `SelfTrainingClassifier` with a logistic-regression
  base. Empty, whitespace-only, `NaN`, and `null` label cells become the `-1`
  sentinel. Split only the labelled rows; append all unlabelled rows to the
  training features; fit the preprocessor on all training rows; evaluate on the
  labelled hold-out. Require at least two classes, 10 labelled rows, and 2
  labelled rows per class.
- **Clustering / anomaly detection:** fit the preprocessor and estimator on all
  rows. Do not create a hold-out split. Compute silhouette, Davies–Bouldin, and
  Calinski–Harabasz for clustering, plus inertia (KMeans/MiniBatchKMeans) or BIC
  (GaussianMixture). For IsolationForest, report `predict` counts and
  `decision_function` moments.

## ONNX notes

- `SelfTrainingClassifier` has no `skl2onnx` converter. Train the wrapper, then
  export the fitted preprocessor + `SelfTrainingClassifier.estimator_`; this
  produces the same predictions because self-training changes only training
  labels. Keep the wrapper in `pkl`/`joblib`.
- IsolationForest requires `target_opset={"": 15, "ai.onnx.ml": 3}`.
  The default ML domain (v4) is rejected by the pinned ONNX Runtime.
- KMeans, MiniBatchKMeans, and GaussianMixture convert with the same opset map.

The full list of shipped estimators (ids, classes, defaults) is documented in [`models.md`](./models.md).
