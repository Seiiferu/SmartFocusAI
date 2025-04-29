# # tests/integration/test_full_flow.py

import cv2
import pytest
from src.logic.focus_manager import FocusManager
from tests.integration.helpers import OfflineTypingDetector, OfflineGazeDetector

@pytest.mark.integration
@pytest.mark.parametrize("fixture,expected", [
    # Sur only_typing.mp4 : on tape → toujours Focused
    ("only_typing.mp4",      ["Focused"] * 150),
    # Sur only_gaze_center.mp4 : pas de frappe, mais regard centré → toujours Focused
    ("only_gaze_center.mp4", ["Focused"] * 150),
    # Sur gaze_away.mp4 : ni frappe ni regard centré → toujours Distracted
    ("gaze_away.mp4",        ["Distracted"] * 150),
])
def test_full_flow(fixture, expected):
    path = f"tests/integration/fixtures/{fixture}"
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"Impossible d'ouvrir {path}"

    # Stubs pour typing/gaze
    typing = OfflineTypingDetector(fixture)
    gaze   = OfflineGazeDetector(thresh=0.35)
    mgr    = FocusManager(typing_detector=typing,
                          gaze_detector=gaze)

    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # note : GazeEstimator attend une frame BGR numpy
        focused = mgr.is_focused(frame)
        results.append("Focused" if focused else "Distracted")

    cap.release()
    assert results == expected
