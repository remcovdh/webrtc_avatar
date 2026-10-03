"""Finding the face in the portrait with MediaPipe (Apache-2.0).

FasterLivePortrait finds and crops the face once, at start-up. By default it
does that with two InsightFace models, whose weights may only be used for
non-commercial research. This module does the same job with Google's
MediaPipe Face Landmarker, whose code and model are Apache-2.0.

FasterLivePortrait derives the crop from the landmarks: the direction from the
eyes to the mouth, and the bounding box of all points. MediaPipe's 478-point
mesh covers the forehead and puts its anchors elsewhere, which gave a crop 15%
wider and lower than the animation model expects. `to_insightface_layout`
therefore returns 106 points that cover the same part of the face as
InsightFace's and have its anchor points in the same places. On five test
portraits the crop then matched InsightFace's within 2% in scale and 9 pixels
(of 512) in position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# MediaPipe face-mesh indices. The contour runs from temple to temple around
# the chin (not over the forehead); with the eyebrows it spans the region
# InsightFace's 106 landmarks span.
_CONTOUR = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
            148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127]
_EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
_INNER = [1, 4, 61, 291, 0, 17, 13, 14, 33, 133, 362, 263]
# Outer corner, inner corner, upper lid, lower lid.
_LEFT_EYE = [33, 133, 159, 145]
_RIGHT_EYE = [362, 263, 386, 374]
_MOUTH_CORNERS = (61, 291)
# Where FasterLivePortrait reads the eyes and the mouth in a 106-point set
# (src/utils/crop.py, parse_pt2_from_pt106).
_SLOTS_LEFT_EYE = [33, 35, 40, 39]
_SLOTS_RIGHT_EYE = [87, 89, 94, 93]
_SLOTS_MOUTH = (52, 61)


def to_insightface_layout(mesh: np.ndarray) -> np.ndarray:
    """478 MediaPipe points (x, y) -> 106 points FasterLivePortrait crops alike."""
    cover = mesh[_CONTOUR + _EYEBROWS + _INNER]
    points = np.array([cover[i % len(cover)] for i in range(106)], dtype=np.float32)
    points[_SLOTS_LEFT_EYE] = mesh[_LEFT_EYE]
    points[_SLOTS_RIGHT_EYE] = mesh[_RIGHT_EYE]
    points[_SLOTS_MOUTH[0]] = mesh[_MOUTH_CORNERS[0]]
    points[_SLOTS_MOUTH[1]] = mesh[_MOUTH_CORNERS[1]]
    return points


class MediaPipeFaceModel:
    """Stands in for FasterLivePortrait's face-analysis model (same `predict`)."""

    def __init__(self, model_path: str, **_ignored: Any) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path,
                    delegate=mp_python.BaseOptions.Delegate.CPU,
                ),
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
        )

    def predict(self, *data: Any) -> list[np.ndarray]:
        """Landmarks per face found in a BGR image; an empty list if none."""
        import cv2
        import mediapipe as mp

        image_bgr = data[0]
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        return [
            to_insightface_layout(
                np.array([[point.x * width, point.y * height] for point in face])
            )
            for face in result.face_landmarks
        ]
