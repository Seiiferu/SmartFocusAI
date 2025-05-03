# # tests/unit/test_combined_gaze_extra.py
# import pytest
# from src.gaze.combined_gaze import CombinedGazeDetector
# import src.gaze.combined_gaze as cg_module

# class FM:
#     def process(self, f):
#         # Renvoie toujours des landmarks non-None
#         return type("FL", (), {"landmark": [1,2,3,4]})()

# class FakeGE:
#     def __init__(self, *args, **kwargs):
#         pass
#     def estimate(self, frame, lm):
#         return "Left"

# def test_combined_gaze_false_when_not_center(monkeypatch):
#     # Stub de GazeEstimator pour qu'il retourne toujours "Left"
#     monkeypatch.setattr(cg_module, "GazeEstimator", FakeGE)
#     det = CombinedGazeDetector(face_mesh=FM())
#     assert det.is_gazing(None) is False


# tests/unit/test_typing_activity_extra.py
import pytest
from src.detection.typing_activity import TypingActivityDetector
import src.detection.typing_activity as ta_module


def test_stop_listener(monkeypatch):
    stopped = {"called": False}
    class FakeListener:
        def __init__(self, on_press): pass
        def start(self): pass
        def stop(self):
            stopped["called"] = True

    # Stub du keyboard.Listener utilisé en interne
    monkeypatch.setattr(ta_module.keyboard, "Listener", lambda on_press: FakeListener(on_press))

    det = TypingActivityDetector(display_timeout=1.0)
    det.start()   # initialise self._listener
    det.stop()    # doit appeler FakeListener.stop()
    assert stopped["called"] is True


