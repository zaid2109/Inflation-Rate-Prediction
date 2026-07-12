"""
Picklable meta-learner classes used inside saved model bundles.

These must live in their own module that is only ever *imported*, never
executed as __main__ (e.g. via `python -m src.train`) — a class defined
inside a script run as __main__ gets pickled with module="__main__", which
only resolves correctly in a process whose __main__ happens to be that same
script. Any other process unpickling the bundle (pytest, uvicorn, a notebook)
has a different __main__ and raises AttributeError. Keeping meta-learners
here, and only ever importing this module (never running it directly),
avoids that trap.
"""

import numpy as np


class ConvexBlendMeta:
    """
    Meta-learner: pred = w * col0 + (1 - w) * col1, with w in [0, 1].

    An earlier version used an unconstrained Ridge meta-model, which produced
    coefficients like [1.01, -0.86] (a strongly *negative* weight on XGBoost) —
    a classic symptom of an unstable fit on two highly collinear inputs
    (OOF Linear vs. OOF XGBoost predictions correlate at r~0.81 on this data).
    A train-only stability check (fit meta-weights on the first half of the
    OOF series, evaluate on the second half) showed the unconstrained Ridge's
    error roughly doubles out-of-sample versus a plain convex blend, so the
    non-negative, sum-to-1 constraint is a real generalization fix, not just
    a cosmetic one — it also makes the displayed ensemble weights match what
    the model actually does (previously the negative XGBoost coefficient was
    silently clipped to "0% weight" for display, which was misleading).
    """

    def __init__(self, w: float):
        self.w = float(w)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return self.w * X[:, 0] + (1 - self.w) * X[:, 1]
