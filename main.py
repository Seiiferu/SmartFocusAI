# main.py

import cv2
from src.detection.typing_activity import TypingActivityDetector
from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.blink_detector import BlinkDetector
from src.logic.focus_manager import FocusManager
from src.gaze.face_mesh import FaceMeshDetector
# (et si tu en as un, ton ObjectActivityDetector, AudioActivityDetector, etc.)

def main():
    cap = cv2.VideoCapture(0)
    typing = TypingActivityDetector(display_timeout=0.5)
    typing.start()
    gaze   = GazeEstimator(thresh=0.35)
    blink  = BlinkDetector()

    focus_mgr = FocusManager(typing, gaze)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mets à jour blink si besoin
            lm = FaceMeshDetector.process(frame)
            if lm:
                blink.update(frame, lm, FaceMeshDetector)

            # Décisions
            focused = focus_mgr.is_focused(frame)
            status  = "Focused" if focused else "Distracted"

            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0) if focused else (0,0,255), 2)
            cv2.imshow("Smart Focus AI", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    finally:
        typing.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
