"""Unit tests for the ensemble meta-learner."""

import pickle
import numpy as np
from src.meta_models import ConvexBlendMeta


def test_convex_blend_predict():
    meta = ConvexBlendMeta(w=0.7)
    X = np.array([[10.0, 0.0], [0.0, 10.0], [4.0, 6.0]])
    preds = meta.predict(X)
    expected = np.array([7.0, 3.0, 0.7 * 4.0 + 0.3 * 6.0])
    np.testing.assert_allclose(preds, expected)


def test_convex_blend_extremes():
    all_linear = ConvexBlendMeta(w=1.0)
    all_xgb = ConvexBlendMeta(w=0.0)
    X = np.array([[5.0, 9.0]])
    assert all_linear.predict(X)[0] == 5.0
    assert all_xgb.predict(X)[0] == 9.0


def test_convex_blend_picklable_across_contexts():
    """Regression test: an earlier version of this class lived in src/train.py
    and was pickled with module="__main__" when training was invoked via
    `python -m src.train`, so any *other* process (pytest, uvicorn) unpickling
    the saved ensemble bundle raised AttributeError. Living in its own
    always-imported module (never run as __main__) avoids that."""
    meta = ConvexBlendMeta(w=0.42)
    blob = pickle.dumps(meta)
    restored = pickle.loads(blob)
    assert restored.w == 0.42
    assert "meta_models" in ConvexBlendMeta.__module__
