"""MLP-specific chart data (neural nets).

MLP jobs keep their task family's charts (C1/C2 or R1/R2/R3 plus M1) and add a
per-epoch training-loss line. See
``docs/07-ui-ux-design/system/charts.md`` §3.8 (MLP1).
"""

from typing import Any

from sklearn.pipeline import Pipeline

MAX_EPOCHS = 5000


def _mlp_progress(pipeline: Pipeline) -> dict[str, Any] | None:
    """Return per-epoch loss for an MLP, or ``None`` when unavailable.

    ``loss_curve_`` is populated for the ``adam`` and ``sgd`` solvers and is
    ``None`` for ``lbfgs`` (and for estimators that are not MLPs), in which case
    the chart is hidden — the same conditional pattern as RL4 and K3.
    """
    estimator = pipeline.named_steps.get('estimator')
    loss_curve = getattr(estimator, 'loss_curve_', None)
    if not loss_curve:
        return None
    losses = [float(value) for value in loss_curve][:MAX_EPOCHS]
    return {
        'epoch': list(range(1, len(losses) + 1)),
        'loss': losses,
    }
