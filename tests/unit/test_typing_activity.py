import time
import pytest
import time
from src.detection.typing_activity import TypingActivityDetector
import src.detection.typing_activity as ta_module

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

def test_typing_activity_detects_typing(monkeypatch):
    det = TypingActivityDetector(display_timeout=1.0)
    # Simule un appui de touche
    det._on_key_press(key=None)
    assert det.is_typing() is True

def test_typing_activity_after_timeout(monkeypatch):
    det = TypingActivityDetector(display_timeout=0.01)
    # Aucun appui
    time.sleep(0.02)
    assert det.is_typing() is False

# def test_typing_after_timeout():
#     det = TypingActivityDetector(display_timeout=0.01)
#     # sans appel à _on_key_press, après le timeout on n'est plus en train de taper
#     time.sleep(0.02)
#     assert det.is_typing() is False

def test_stop_listener(monkeypatch):
    stopped = {"called": False}
    class FakeListener:
        def __init__(self, on_press):
            pass
        def start(self):
            pass
        def stop(self):
            stopped["called"] = True

    # Stub du constructeur keyboard.Listener dans le module
    monkeypatch.setattr(ta_module.keyboard, "Listener", lambda on_press: FakeListener(on_press))

    det = TypingActivityDetector(display_timeout=1.0)
    det.start()   # initialise self._listener
    det.stop()    # doit appeler FakeListener.stop()
    assert stopped["called"] is True