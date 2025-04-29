# app.py

import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from src.detection.typing_activity import TypingActivityDetector
from src.gaze.gaze_estimator import GazeEstimator

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Instancie tes détecteurs
        self.typing = TypingActivityDetector(display_timeout=0.5)
        self.gaze   = GazeEstimator(thresh=0.35)
        self.typing.start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Tes détections
        is_typing = self.typing.is_typing()
        is_gazing = self.gaze.is_gazing(img)

        status = "Focused" if (is_typing or is_gazing) else "Distracted"
        color  = (0,255,0) if status=="Focused" else (0,0,255)

        # Overlay
        cv2.putText(img, status, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        self.typing.stop()

def main():
    st.title("Smart Focus AI Demo")
    st.write("Autorisez l'accès à votre webcam pour lancer la démo.")
    webrtc_streamer(
        key="focus-demo",
        mode="SENDRECV",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

if __name__ == "__main__":
    main()
