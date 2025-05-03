import pytest
from src.gaze.smoother import DirectionSmoother

def test_initial_state():
    sm = DirectionSmoother(window_size=3)
    assert sm.current == "None"

def test_update_needs_majority():
    sm = DirectionSmoother(window_size=3)
    # 1er update → 1 entrée suffit pour passer à "Left"
    assert sm.update("Left") == "Left"
    assert sm.current == "Left"
    # un autre sens ne passe pas tant qu’il n’est pas majoritaire
    assert sm.update("Right") == "Left"
    assert sm.update("Right") == "Right"
