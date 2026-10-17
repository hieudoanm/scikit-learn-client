# skl — Neural networks (tabular MLPs)

The neural-net offering is scikit-learn's `MLPClassifier` / `MLPRegressor`,
exposed as the catalogue entries `mlp_classifier` (task `classification`) and
`mlp_regressor` (task `regression`). They are multi-layer perceptrons for
**tabular** data — depth, width, activation, solver, regularisation, and early
stopping are all configurable.

Riding on the scikit-learn estimator keeps a single `Pipeline` + `skl2onnx`
export path and needs no new dependency.

## Hyperparameters

Both entries share the same knobs (defaults form the shipped baseline):

| Hyperparameter        | Type            | Default        | Notes                                                        |
| --------------------- | --------------- | -------------- | ------------------------------------------------------------ |
| `hidden_layer_sizes`  | `int[]`         | `[100]`        | One integer per hidden layer; `[]` means no hidden layer     |
| `activation`          | string          | `"relu"`       | `identity`, `logistic`, `tanh`, `relu`                       |
| `solver`              | string          | `"adam"`       | `lbfgs`, `sgd`, `adam`                                       |
| `alpha`               | number          | `0.0001`       | L2 penalty; regularises weights                              |
| `learning_rate_init`  | number          | `0.001`        | Initial step size for `sgd` / `adam`                         |
| `max_iter`            | integer         | `300`          | Maximum epochs                                               |
| `batch_size`          | integer / auto  | `"auto"`       | `"auto"` = `min(200, n_samples)`                             |
| `early_stopping`      | boolean         | `false`        | Hold out `validation_fraction` to stop on no improvement     |
| `validation_fraction` | number          | `0.1`          | Fraction held out when `early_stopping=true`                 |
| `n_iter_no_change`    | integer         | `10`           | Patience for early stopping / `tol`                          |
| `tol`                 | number          | `0.0001`       | Optimisation tolerance                                       |
| `shuffle`             | boolean         | `true`         | Shuffle samples each epoch (`sgd` / `adam`)                  |

`random_state` is fixed to the job seed (default 42) so fits are reproducible.

## Validation (FR-027)

Out-of-range values are rejected with `INVALID_HYPERPARAMETERS`:

- `hidden_layer_sizes` — a list of 1–5 integers, each ≥ 1.
- `activation` — one of `identity`, `logistic`, `tanh`, `relu`.
- `solver` — one of `lbfgs`, `sgd`, `adam`.
- `alpha` — number ≥ 0.
- `learning_rate_init` — number in `(0, 1]`.
- `max_iter` — integer ≥ 1.
- `batch_size` — `"auto"` or an integer ≥ 1.
- `validation_fraction` — number in `(0, 1)`.
- `early_stopping` — boolean.

Unknown keys are rejected. `n_iter_no_change`, `tol`, `shuffle`, and the
remaining sklearn parameters pass through to the estimator unvalidated.

## Export

Because both entries are ordinary scikit-learn estimators, the fitted pipeline
(preprocessor + MLP) converts to ONNX directly with `skl2onnx` — the same path
as every other supervised entry. The parity test
(`tests/training/core/test_pipeline_export.py`) fits, scores, exports, and
compares ONNX Runtime predictions against scikit-learn for both entries.

## Charts

MLP jobs get their task family's charts (C1/C2 for classification, R1/R2/R3 for
regression; M1 in both), plus:

- **M1 feature importance** is computed from the **first hidden layer**
  (`coefs_[0]`): the mean absolute outgoing weight per input, aggregated back to
  the original feature columns. `method` is `mlp_weights`. This replaces the
  slow permutation fallback for MLPs.
- **MLP1 training loss** — a per-epoch line from `loss_curve_`, present only for
  `solver` of `adam` or `sgd` (`lbfgs` leaves `loss_curve_` as `None`, so the
  card is hidden).

See [`charts.md`](../../07-ui-ux-design/system/charts.md#38-neural-nets-mlp-classification-and-regression).

## Notes

- MLP fitting is iterative and can be slow and non-deterministic across
  threads; the job seed is fixed for reproducibility, and a convergence warning
  is expected on small or hard datasets (raise `max_iter` or enable
  `early_stopping`).
- Standardising inputs (the shared preprocessor) matters more for MLPs than for
  tree ensembles.
