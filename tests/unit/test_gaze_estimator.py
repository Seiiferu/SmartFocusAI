# # tests/unit/test_gaze_estimator.py

import numpy as np
import pytest
from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector

class DummyLM:
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y

def test_center_detection_with_thresh(monkeypatch):
    # 1) Stub indices pour éviter tout index hors-liste
    monkeypatch.setattr(BlinkDetector, "LEFT_EYE_IDX",  [0, 1, 2, 3])
    monkeypatch.setattr(BlinkDetector, "RIGHT_EYE_IDX", [0, 1, 2, 3])
    monkeypatch.setattr(GazeEstimator, "LEFT_IRIS_IDXS",  [0, 1, 2, 3])
    monkeypatch.setattr(GazeEstimator, "RIGHT_IRIS_IDXS", [0, 1, 2, 3])

    # 2) Stub norm pour rendre raw = 0 sans division par zéro
    monkeypatch.setattr(np.linalg, "norm", lambda x: 1.0)

    # 3) Fake MultiFaceLandmarks avec 4 points identiques
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})

    # Frame factice
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Legacy API : thunk centre-only
    ge = GazeEstimator(thresh=0.5, axis=0, momentum=0.0)

    # raw calculé sera 0.0 → tombe dans (-0.5, 0.5) → "Center"
    gaze = ge.estimate(frame, FL)
    assert gaze == "Center"
