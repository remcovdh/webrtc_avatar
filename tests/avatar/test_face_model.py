"""Tests for the MediaPipe face landmarks in FasterLivePortrait's layout."""

import unittest

import numpy as np

from avatar import face_model
from avatar.face_model import to_insightface_layout


def mesh() -> np.ndarray:
    """478 points whose coordinates say which point they are."""
    return np.array([[index, index + 1000] for index in range(478)], dtype=np.float32)


class LayoutTests(unittest.TestCase):
    def test_anchor_slots_hold_the_eyes_and_mouth_corners(self) -> None:
        points = to_insightface_layout(mesh())
        self.assertEqual(points.shape, (106, 2))
        # The slots FasterLivePortrait averages for the eye centres ...
        self.assertEqual(points[[33, 35, 40, 39], 0].tolist(), [33, 133, 159, 145])
        self.assertEqual(points[[87, 89, 94, 93], 0].tolist(), [362, 263, 386, 374])
        # ... and for the mouth centre.
        self.assertEqual(points[[52, 61], 0].tolist(), [61, 291])

    def test_no_forehead_point_widens_the_bounding_box(self) -> None:
        forehead = {10, 338, 297, 332, 284, 251, 389, 109, 67, 103, 54, 21, 162}
        used = set(to_insightface_layout(mesh())[:, 0].astype(int).tolist())
        self.assertEqual(used & forehead, set())
        # Chin, both sides of the face and the eyebrows are covered.
        self.assertLessEqual({152, 234, 454, 105, 334}, used)

    def test_crop_anchors_follow_flp_definition(self) -> None:
        """Eye centre and mouth centre as FasterLivePortrait computes them."""
        rng = np.random.default_rng(0)
        points478 = rng.random((478, 2)).astype(np.float32) * 500
        points = to_insightface_layout(points478)
        left = points[[33, 35, 40, 39]].mean(axis=0)
        right = points[[87, 89, 94, 93]].mean(axis=0)
        np.testing.assert_allclose(left, points478[face_model._LEFT_EYE].mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(right, points478[face_model._RIGHT_EYE].mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(
            (points[52] + points[61]) / 2, (points478[61] + points478[291]) / 2, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
