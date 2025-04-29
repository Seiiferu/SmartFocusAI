# # main.py

# import time
# import cv2
# from src.detection.typing_activity import TypingActivityDetector
# from src.gaze.gaze_estimator import GazeEstimator
# from src.gaze.blink_detector import BlinkDetector  # ton module blink

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Erreur : impossible d'ouvrir la caméra.")
#         return

#     typing_det = TypingActivityDetector(display_timeout=0.5)
#     typing_det.start()

#     gaze_det  = GazeEstimator(thresh=0.35)
#     blink_det = BlinkDetector(threshold=0.2, consecutive_frames=2)

#     last_blink_time = 0.0
#     BLINK_MASK_DURATION = 0.5  # secondes pendant lesquelles on ignore le gaze après un blink

#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             now = time.time()

#             # 1) Détection blink
#             is_blink = blink_det.detect(frame)
#             if is_blink:
#                 last_blink_time = now

#             # 2) Détection typing et gaze (si pas trop récemment en blink)
#             is_typing = typing_det.is_typing()
#             gaze_allowed = (now - last_blink_time) > BLINK_MASK_DURATION
#             gaze_on = gaze_allowed and gaze_det.is_gazing(frame)

#             # 3) Logique Focus / Distracted
#             #    - Si on tape, on est concentré quel que soit le gaze
#             #    - Sinon, on est concentré si on regarde l’écran (après blink masking)
#             if is_typing or gaze_on:
#                 status, color = "Focused", (0,255,0)
#             else:
#                 status, color = "Distracted", (0,0,255)

#             # 4) Affichage
#             cv2.putText(frame, status, (10,30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#             # Afficher un carré de feedback blink/typing/gaze
#             cv2.rectangle(frame, (10,50), (120,100), 
#                           (0,0,255) if is_blink else (0,255,0), 2)
#             cv2.putText(frame, "BLINK" if is_blink else "", (15,90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

#             cv2.imshow("Smart Focus AI", frame)

#             if cv2.waitKey(30) & 0xFF == ord('q'):
#                 break

#     finally:
#         typing_det.stop()
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
