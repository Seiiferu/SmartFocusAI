# src/logic/focus_manager.py

from src.detection.typing_activity import TypingActivityDetector
from src.gaze.gaze_estimator import GazeEstimator

class FocusManager:
    """
    Décide si l'utilisateur est Focused ou Distracted selon :
      1) Si on tape sur le clavier → Focused (override quel que soit le gaze)
      2) Sinon, si le regard est centré → Focused
      3) Sinon → Distracted
    """

    def __init__(self,
                 typing_detector: TypingActivityDetector,
                 gaze_detector: GazeEstimator):
        self.typing = typing_detector
        self.gaze   = gaze_detector

    def is_focused(self, frame) -> bool:
        # 1) priorité au clavier
        is_typing = self.typing.is_typing()
        if is_typing:
            # même si le regard est hors-centre, on reste concentré
            return True

        # 2) si on ne tape pas, on regarde le gaze
        is_gazing = self.gaze.is_gazing(frame)
        if is_gazing:
            # regard centré ET pas de frappe → on est concentré
            return True

        # 3) ni frappe ni regard centré → distrait
        return False
