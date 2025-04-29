# tests/unit/test_blink_activity.py

import numpy as np
import pytest
from src.gaze.blink_detector import BlinkDetector

def test_eye_aspect_ratio_open_eye():
    """
    Un œil « ouvert » doit donner un EAR relativement élevé (> 0.3).
    """
    # Construit 6 points formant un œil ouvert
    pts = np.array([
        [0.0, 0.0],   # coin gauche
        [1.0, 1.0],   # haut gauche
        [2.0, 1.0],   # haut droite
        [3.0, 0.0],   # coin droit
        [2.0, -1.0],  # bas droite
        [1.0, -1.0],  # bas gauche
    ], dtype=float)

    ear = BlinkDetector.eye_aspect_ratio(pts)
    # calcul manuel attendu
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    expected = (A + B) / (2.0 * C)
    assert ear == pytest.approx(expected, rel=1e-6)
    assert ear > 0.3

def test_eye_aspect_ratio_closed_eye():
    """
    Un œil presque fermé doit donner un EAR très faible (< 0.1).
    """
    pts = np.array([
        [0.0, 0.0],
        [1.0, 0.01],
        [2.0, 0.01],
        [3.0, 0.0],
        [2.0, -0.01],
        [1.0, -0.01],
    ], dtype=float)

    ear = BlinkDetector.eye_aspect_ratio(pts)
    assert ear < 0.1

class FakeBlinkDetector(BlinkDetector):
    """
    Override pour :
      - Utiliser seulement 6 indices (0..5) au lieu de la liste
      - Injecter une séquence d'EAR contrôlées
    """
    LEFT_EYE_IDX = [0,1,2,3,4,5]

    def __init__(self, ear_sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ears = list(ear_sequence)

    def eye_aspect_ratio(self, pts):
        # Retourne la prochaine valeur de la séquence
        return self._ears.pop(0)

def make_dummy_fm_and_landmarks():
    # Dummy FM : landmark_to_pixel renvoie (0,0) quel que soit le point
    class DummyFM:
        def landmark_to_pixel(self, frame, lm):
            return np.array([0.0, 0.0], dtype=float)
    # Dummy landmarks : liste de 6 placeholders
    class DummyLM:
        landmark = [None]*6
    return DummyFM(), DummyLM()

def test_blink_counting_logic():
    """
    Simule 2 frames « fermées » puis 1 frame « ouverte » :
    - COUNTER passe à 2
    - puis OPEN incrémente blink_count à 1 et remet counter à 0
    """
    # sequence d'EAR : 0.1 (< seuil), 0.1 (< seuil), 0.3 (> seuil)
    seq = [0.1, 0.1, 0.3]
    bd = FakeBlinkDetector(ear_sequence=seq)
    fm, lm = make_dummy_fm_and_landmarks()

    # 1er appel : oeil « fermé »
    c1 = bd.update(frame=None, landmarks=lm, fm=fm)
    assert c1 == 0
    assert bd.counter == 1

    # 2e appel : encore fermé
    c2 = bd.update(frame=None, landmarks=lm, fm=fm)
    assert c2 == 0
    assert bd.counter == 2

    # 3e appel : oreil « ouvert », et counter>=CONSEC_FRAMES → blink_count=1
    c3 = bd.update(frame=None, landmarks=lm, fm=fm)
    assert c3 == 1
    assert bd.counter == 0

    # Si on referme et rouvre encore, on obtient un 2e clignement
    bd._ears = [0.05, 0.05, 0.5]
    c4 = bd.update(frame=None, landmarks=lm, fm=fm)  # fermé
    c5 = bd.update(frame=None, landmarks=lm, fm=fm)  # fermé
    c6 = bd.update(frame=None, landmarks=lm, fm=fm)  # ouvert
    assert c6 == 2
