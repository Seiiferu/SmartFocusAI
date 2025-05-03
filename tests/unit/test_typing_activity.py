import time
import pytest
from src.detection.typing_activity import TypingActivityDetector

# On injecte un Listener factice pour éviter l'erreur X11
class DummyListener:
    def __init__(self, on_press): pass
    def start(self): pass
    def stop(self): pass

@pytest.fixture(autouse=True)
def patch_listener(monkeypatch):
    import src.detection.typing_activity as mod
    monkeypatch.setattr(mod.keyboard, "Listener", DummyListener)

def test_is_typing_initial_false():
    det = TypingActivityDetector(display_timeout=0.1)
    assert det.is_typing() is False

def test_is_typing_after_key():
    det = TypingActivityDetector(display_timeout=0.1)
    # on simule une frappe
    det._last_event = time.time()
    assert det.is_typing() is True
