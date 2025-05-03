# # tests/unit/test_gaze_estimator.py

import numpy as np
import pytest
from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector

class DummyLM:
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y

@pytest.fixture(autouse=True)
def stub_indices_and_norm(monkeypatch):
    # stub des listes d’indices pour éviter tout hors-liste
    monkeypatch.setattr(BlinkDetector, "LEFT_EYE_IDX",  [0,1,2,3])
    monkeypatch.setattr(BlinkDetector, "RIGHT_EYE_IDX", [0,1,2,3])
    monkeypatch.setattr(GazeEstimator, "LEFT_IRIS_IDXS",  [0,1,2,3])
    monkeypatch.setattr(GazeEstimator, "RIGHT_IRIS_IDXS", [0,1,2,3])
    # stub de la norme pour que la division donne toujours 0/1 ou 1/1
    monkeypatch.setattr(np.linalg, "norm", lambda v: 1.0)

def test_estimate_multiple_candidates():
    # ranges qui se chevauchent : A et B couvrent raw=0
    ge = GazeEstimator(calibration_ranges={"A":(-1,1),"B":(-2,2)}, axis=0, momentum=0.0)
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})
    frame = np.zeros((5,5,3), dtype=np.uint8)
    gaze = ge.estimate(frame, FL)
    # raw = 0 -> candidat A et B -> on choisit celui dont le centre est le plus proche de 0
    assert gaze in ("A","B")

def test_smoothing_momentum():
    ge = GazeEstimator(thresh=1.0, axis=0, momentum=0.5)
    FL = type("FL", (), {"landmark": [DummyLM() for _ in range(4)]})
    frame = np.zeros((5,5,3), dtype=np.uint8)
    # premier appel : raw=0 -> hist=0
    _ = ge.estimate(frame, FL)
    # stub raw à 1 en remplaçant _to_px pour forcer vec=1
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(GazeEstimator, "_to_px", lambda self, f, lm: np.array((1.0,0.0)))
    gaze = ge.estimate(frame, FL)
    # hist = 0*0.5 + 1*0.5 = 0.5 -> centre seul en legacy -> "Center"
    assert gaze == "Center"
    monkeypatch.undo()