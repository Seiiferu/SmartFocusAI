import numpy as np
import pytest
from src.gaze.face_mesh import FaceMeshDetector

# fixture qui simule MediaPipe face_mesh
class DummyFaceMesh:
    def __init__(self, **kwargs): pass
    def process(self, rgb):
        class R: multi_face_landmarks = []
        return R()

@pytest.fixture(autouse=True)
def patch_mediapipe(monkeypatch):
    import src.gaze.face_mesh as fm
    # on crée un objet solutions.face_mesh
    fake_mod = type("M", (), {})()
    fake_mod.FaceMesh = DummyFaceMesh
    mp = type("MP", (), {"solutions": type("S", (), {"face_mesh": fake_mod})})()
    monkeypatch.setattr(fm, "mp", mp)

def test_process_returns_none_if_empty():
    detector = FaceMeshDetector()
    result = detector.process(np.zeros((100,100,3), dtype=np.uint8))
    assert result is None


def test_face_mesh_detects_landmarks(monkeypatch):
    # Simule un résultat MediaPipe avec un seul visage
    class Dummy:
        multi_face_landmarks = [object()]
    fake_mp = type("M", (), {
        "solutions": type("S", (), {"face_mesh": type("FM", (), {"FaceMesh": lambda *a,**k: DummyFM()})})
    })
