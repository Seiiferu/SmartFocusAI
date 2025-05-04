# tests/unit/test_gaze_estimator_extra.py

import numpy as np
import pytest

from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector

def test_is_gazing_api(monkeypatch):
    # Mode legacy (thresh) pour simplifier
    ge = GazeEstimator(thresh=0.1, axis=0)

    # 1) pas de visage détecté → False
    monkeypatch.setattr(ge.fm, "process", lambda frame: None)
    assert ge.is_gazing(None) is False

    # 2) visage détecté mais estimate ≠ "Center" → False
    class FL: pass
    monkeypatch.setattr(ge.fm, "process", lambda frame: FL())
    monkeypatch.setattr(ge, "estimate", lambda frame, lm: "Left")
    assert ge.is_gazing(None) is False

    # 3) visage + estimate "Center" → True
    monkeypatch.setattr(ge, "estimate", lambda frame, lm: "Center")
    assert ge.is_gazing(None) is True
