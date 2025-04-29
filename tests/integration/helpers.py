import cv2
from src.gaze.gaze_estimator import GazeEstimator

class OfflineTypingDetector:
    """
    Stub : déduit is_typing() du nom de la fixture.
    """
    def __init__(self, fixture_name: str):
        # si le nom contient "typing", on renvoie toujours True
        self._typing = "typing" in fixture_name

    def is_typing(self) -> bool:
        return self._typing

class OfflineGazeDetector(GazeEstimator):
    """
    Hérite de ta vraie GazeEstimator.
    On l'utilise frame par frame sur la vidéo.
    """
    def __init__(self, thresh=0.35):
        super().__init__(thresh=thresh)

    def is_gazing(self, frame) -> bool:
        # réutilise la méthode parent pour vidéo
        return super().is_gazing(frame)
