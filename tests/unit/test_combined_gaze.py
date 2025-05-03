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