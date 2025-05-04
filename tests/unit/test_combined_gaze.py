# tests/unit/test_combined_gaze.py

import pytest
import src.gaze.combined_gaze as cg
from src.gaze.combined_gaze import CombinedGazeDetector


class DummyFM:
    def process(self, frame):
        # Retourne un fake landmarks non-None
        return object()
    
# GazeEstimator factice pour renvoyer "Center"
class FakeGE:
    def __init__(self, *a, **k): pass
    def estimate(self, frame, lm): return "Center"

# GazeEstimator factice pour renvoyer "Left"
class FakeGE_Left(FakeGE):
    def estimate(self, frame, lm): return "Left"

def test_combined_gaze_gazing(monkeypatch):
    monkeypatch.setattr(cg, "GazeEstimator", FakeGE)
    det = CombinedGazeDetector(face_mesh=DummyFM())
    assert det.is_gazing(None) is True

def test_combined_gaze_not_center(monkeypatch):
    monkeypatch.setattr(cg, "GazeEstimator", FakeGE_Left)
    det = CombinedGazeDetector(face_mesh=DummyFM())
    assert det.is_gazing(None) is False

def test_combined_gaze_process_none():
    class FMNone:
        def process(self, frame): return None
    det = CombinedGazeDetector(face_mesh=FMNone())
    assert det.is_gazing(None) is False


def test_combined_gaze_two_axes_switch():
    # Instanciation standard
    det = CombinedGazeDetector(face_mesh=DummyFM())

    # On stubbe ge_h et ge_v pour retourner successivement deux valeurs
    seq_h = iter(["Left", "Center"])
    seq_v = iter(["None",  "None"])
    det.ge_h = type("GH", (), {"estimate": lambda self, f, lm: next(seq_h)})()
    det.ge_v = type("GV", (), {"estimate": lambda self, f, lm: next(seq_v)})()

    # On stubbe les smoothers pour qu'ils renvoient exactement la valeur reçue
    det.h_smoother = type("SH", (), {"update": lambda self, d: d})()
    det.v_smoother = type("SV", (), {"update": lambda self, d: d})()

    # 1er appel : ge_h="Left", ge_v="None" → ni l’un ni l’autre "Center" → False
    assert det.is_gazing(None) is False

    # 2ème appel : ge_h="Center", ge_v="None" → h_smoother="Center" → True
    assert det.is_gazing(None) is True