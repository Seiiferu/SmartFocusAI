#tests/unit/test_combined_gaze.py

from src.gaze.combined_gaze import CombinedGazeDetector
import src.gaze.combined_gaze as cg

class DummyFM:
    def process(self, frame): return ["any"]

class DummyGE:
    def __init__(self,*a,**k): pass
    def estimate(self, f, lm): return "Center"

class DummyDS:
    def __init__(self,*a,**k): self.current="None"
    def update(self, d): 
        self.current = d
        return d

def test_combined_gaze_gazing(monkeypatch):
    import src.gaze.combined_gaze as cg
    monkeypatch.setattr(cg, "FaceMeshDetector", lambda *a,**k: DummyFM())
    monkeypatch.setattr(cg, "GazeEstimator", lambda *a,**k: DummyGE())
    monkeypatch.setattr(cg, "DirectionSmoother", lambda *a,**k: DummyDS())
    
    # on crée nous-même le DummyFM et on le passe
    fm = DummyFM()
    det = cg.CombinedGazeDetector(face_mesh=fm)
    assert det.is_gazing(None) is True

def test_combined_gaze_false_when_not_center(monkeypatch):
    # Stubber d’un FaceMeshDetector qui renvoie des landmarks quelconques
    class FM:
        def process(self, f): 
            return type("FL", (), {"landmark": [1,2,3,4]})()
    det = CombinedGazeDetector(face_mesh=FM())
    # Stubber estimate pour renvoyer autre chose que "Center"
    monkeypatch.setattr(det, "estimate", lambda frame, lm: "Left")
    assert det.is_gazing(None) is False

def test_combined_gaze_false_when_not_center(monkeypatch):
    # Stub de FaceMeshDetector
    class FM:
        def process(self, f): 
            return type("FL", (), {"landmark": [1,2,3,4]})()

    # FakeGazeEstimator pour forcer un retour != "Center"
    class FakeGE:
        def __init__(self, *a, **k): pass
        def estimate(self, frame, lm): return "Left"

    # On patche la classe GazeEstimator **dans** le module
    monkeypatch.setattr(cg, "GazeEstimator", FakeGE)

    det = cg.CombinedGazeDetector(face_mesh=FM())
    assert det.is_gazing(None) is False
