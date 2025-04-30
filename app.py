# # app.py

# import streamlit as st
# import av
# import cv2
# import numpy as np
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# from src.detection.typing_activity import TypingActivityDetector
# from src.gaze.gaze_estimator import GazeEstimator

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
# )

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         # Instancie tes détecteurs
#         self.typing = TypingActivityDetector(display_timeout=0.5)
#         self.gaze   = GazeEstimator(thresh=0.35)
#         self.typing.start()

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="bgr24")

#         # Tes détections
#         is_typing = self.typing.is_typing()
#         is_gazing = self.gaze.is_gazing(img)

#         status = "Focused" if (is_typing or is_gazing) else "Distracted"
#         color  = (0,255,0) if status=="Focused" else (0,0,255)

#         # Overlay
#         cv2.putText(img, status, (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")

#     def __del__(self):
#         self.typing.stop()

# def main():
#     st.title("Smart Focus AI Demo")
#     st.write("Autorisez l'accès à votre webcam pour lancer la démo.")
#     webrtc_streamer(
#         key="focus-demo",
#         mode="SENDRECV",
#         rtc_configuration=RTC_CONFIGURATION,
#         video_processor_factory=VideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#     )

# if __name__ == "__main__":
#     main()


# app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import threading
import time

from src.detection.typing_activity import TypingActivityDetector
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector
# from src.gaze.gaze_estimator import GazeEstimator
from src.gaze.combined_gaze import CombinedGazeDetector
from src.gaze.smoother import DirectionSmoother
from src.logic.focus_manager import FocusManager

st.set_page_config(page_title="Smart Focus AI", layout="wide")

class FocusTransformer(VideoTransformerBase):
    def __init__(self):
        # démarrage des détecteurs
        self.typing = TypingActivityDetector(display_timeout=0.5)
        self.typing.start()
        self.face_mesh = FaceMeshDetector()
        self.blink = BlinkDetector()
        self.gaze = CombinedGazeDetector(self.face_mesh)
        self.focus_mgr = FocusManager(typing_detector=self.typing,
                                      gaze_detector=self.gaze)

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # blink
        lm = self.face_mesh.process(img)
        if lm is not None:
            self.blink.update(img, lm, self.face_mesh)

        # typing & gaze & focus
        is_typing = self.typing.is_typing()
        is_gazing = self.gaze.is_gazing(img)
        focused   = self.focus_mgr.is_focused(img)

        # overlay
        status = "Focused" if focused else "Distracted"
        color  = (0,255,0) if focused else (0,0,255)
        cv2.putText(img, status, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Typing:{int(is_typing)} Center Gaze:{int(is_gazing)} Blink:{self.blink.blink_count}",
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🎯 Smart Focus AI")
st.write("Autorisez la webcam ci-dessous puis appuyez sur *Start*.")

webrtc_streamer(
    key="smart-focus",
    video_processor_factory=FocusTransformer,      
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True                          
)

