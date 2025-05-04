# # tests/unit/test_focus_manager.py
# import pytest
# import src.logic.focus_manager as fm
# import src.gaze.combined_gaze as cg_module

# class FakeDetector:
#     def __init__(self, *args, **kwargs):
#         pass
#     def is_gazing(self, frame):
#         return True
#     def is_typing(self):
#         return False

# @pytest.fixture(autouse=True)
# def patch_detectors(monkeypatch):
#     # Stubber TypingActivityDetector dans focus_manager
#     monkeypatch.setattr(fm, "TypingActivityDetector", FakeDetector)
#     # Stubber CombinedGazeDetector **là où il est réellement défini**
#     monkeypatch.setattr(cg_module, "CombinedGazeDetector", FakeDetector)

# def test_focus_manager_switch():
#     # On fournit explicitement nos stubs aux deux arguments
#     typing_stub = FakeDetector()
#     gaze_stub   = FakeDetector()
#     manager = fm.FocusManager(typing_stub, gaze_stub)
#     # comme FakeDetector.is_typing()->False, is_gazing()->True
#     assert manager.is_focused(None) is True


import pytest
from src.logic.focus_manager import FocusManager

class FakeDetector:
    def __init__(self, *a, **k): pass
    def is_typing(self): return False
    def is_gazing(self, frame): return True

def test_focus_manager_switch():
    td = FakeDetector()
    gd = FakeDetector()
    mgr = FocusManager(typing_detector=td, gaze_detector=gd)
    assert mgr.is_focused(None) is True
