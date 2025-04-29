# tests/unit/test_focus_manager.py

import pytest
from src.logic.focus_manager import FocusManager

# Dummy simple qui expose la même interface que tes vrais detectors
class DummyDetector:
    def __init__(self, value: bool):
        self._value = value
    def is_typing(self) -> bool:
        return self._value
    def is_gazing(self, frame) -> bool:
        return self._value

@pytest.mark.parametrize("typing,gazing,expected", [
    (True,  True,  True),   # tape ET regarde → focused
    (True,  False, True),   # tape seul        → focused
    (False, True,  True),   # regarde seul     → focused
    (False, False, False),  # ni tape ni regarde → distracted
])
def test_focus_logic(typing, gazing, expected):
    # On crée deux dummy distincts, chacun renvoie soit typing soit gazing
    typing_dummy = DummyDetector(typing)
    gaze_dummy   = DummyDetector(gazing)
    mgr = FocusManager(typing_detector=typing_dummy,
                       gaze_detector=gaze_dummy)
    assert mgr.is_focused(frame=None) == expected
