# src/gaze/gaze_estimator.py

import cv2
import numpy as np
from src.gaze.face_mesh import FaceMeshDetector
from src.gaze.blink_detector import BlinkDetector

class GazeEstimator:
    LEFT_IRIS_IDXS  = FaceMeshDetector.LEFT_IRIS_IDXS
    RIGHT_IRIS_IDXS = FaceMeshDetector.RIGHT_IRIS_IDXS

    def __init__(
        self,
        calibration_ranges: dict,
        momentum: float = 0.8,
        axis: int = 0  # 0 for horizontal (x), 1 for vertical (y)
    ):
        # calibration_ranges: dict direction -> (min, max)
        self.calibration_ranges = calibration_ranges
        self.momentum = momentum
        self.axis = axis
        self._hist = 0.0
        self.fm = FaceMeshDetector()

    @staticmethod
    def _to_px(frame, lm):
        h, w = frame.shape[:2]
        return np.array((lm.x * w, lm.y * h), dtype=np.float32)

    def estimate(self, frame, face_landmarks):
        lm = getattr(face_landmarks, "landmark", None)
        if lm is None:
            return None

        # 1) Eye landmarks for blink/eyelid
        LE, RE = BlinkDetector.LEFT_EYE_IDX, BlinkDetector.RIGHT_EYE_IDX
        eye_pts_L = np.array([self._to_px(frame, lm[i]) for i in LE], np.float32)
        eye_pts_R = np.array([self._to_px(frame, lm[i]) for i in RE], np.float32)
        eye_c_L, eye_c_R = eye_pts_L[[0,3]].mean(0), eye_pts_R[[0,3]].mean(0)

        # 2) Iris landmarks
        iris_pts_L = np.array([self._to_px(frame, lm[i]) for i in self.LEFT_IRIS_IDXS], np.float32)
        iris_pts_R = np.array([self._to_px(frame, lm[i]) for i in self.RIGHT_IRIS_IDXS], np.float32)
        iris_c_L, iris_c_R = iris_pts_L.mean(0), iris_pts_R.mean(0)

        # 3) Displacement vectors normalized by eye width
        vec_L = (iris_c_L - eye_c_L) / np.linalg.norm(eye_pts_L[3] - eye_pts_L[0])
        vec_R = (iris_c_R - eye_c_R) / np.linalg.norm(eye_pts_R[3] - eye_pts_R[0])
        # select axis (0=x, 1=y)
        raw = float((vec_L[self.axis] + vec_R[self.axis]) / 2.0)

        # 4) Smoothing
        self._hist = self._hist * self.momentum + raw * (1.0 - self.momentum)

        # debug console & visual
        print(f"[DEBUG axis={self.axis}] raw={raw:.2f} hist={self._hist:.2f}")
        cv2.putText(
            frame,
            f"r={raw:.2f} h={self._hist:.2f}",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        # 5) Classification
        candidates = [d for d, (mn, mx) in self.calibration_ranges.items()
                      if mn <= self._hist <= mx]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # resolve overlaps by nearest interval center
        centers = {d: (self.calibration_ranges[d][0] + self.calibration_ranges[d][1]) / 2
                   for d in candidates}
        best = min(candidates, key=lambda d: abs(self._hist - centers[d]))
        return best
