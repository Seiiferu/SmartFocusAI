# tests/integration/test_full_flow.py

import os
import cv2
import pytest

from tests.integration.helpers import OfflineTypingDetector, OfflineGazeDetector
from src.logic.focus_manager import FocusManager

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WARMUP_CALLS = 11  # window_size 10 → 11 appels pour sortir du warm-up

@pytest.mark.parametrize(
    "fixture_name, typing_flag, expected_focus",
    [
        # Typing override (toujours True, même avec image)
        (None,         True,  True),
        ("center.jpg", True,  True),
        ("left.jpg",   True,  True),
        ("right.jpg",  True,  True),

        # Pas de frappe → on regarde le gaze
        ("center.jpg", False, True),   # regard centré → Focused
        ("left.jpg",   False, False),  # regard à gauche → Distracted
        ("right.jpg",  False, False),  # regard à droite → Distracted
    ]
)
def test_full_flow(fixture_name, typing_flag, expected_focus):
    # Création des détecteurs
    typing_detector = OfflineTypingDetector("typing" if typing_flag else "no_typing")
    gaze_detector   = OfflineGazeDetector(fixture_name)
    fm = FocusManager(typing_detector, gaze_detector)

    # Préparer le frame (None en cas de typing-only)
    if fixture_name is None:
        frame = None
    else:
        path = os.path.join(FIXTURE_DIR, fixture_name)
        assert os.path.exists(path), f"Fichier introuvable : {path}"
        frame = cv2.imread(path)
        assert frame is not None, f"Impossible de lire l'image : {path}"

    # Mini warm-up (window_size = 10)
    for _ in range(WARMUP_CALLS):
        fm.is_focused(frame)

    # Vérification finale
    assert fm.is_focused(frame) is expected_focus
