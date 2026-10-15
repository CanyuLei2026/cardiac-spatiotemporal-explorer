import unittest

import numpy as np

from training.data import CaseArrays, activation_frames, make_examples


class TrainingDataTests(unittest.TestCase):
    def synthetic_case(self) -> CaseArrays:
        frame_count = 140
        point_count = 8
        signals = np.zeros((point_count, frame_count), dtype=np.float32)
        for point_index in range(point_count):
            activation = 75 + point_index
            signals[point_index, activation:] = 1.0
        points = np.arange(point_count * 3, dtype=np.float32).reshape(point_count, 3)
        return CaseArrays("synthetic", points, signals, np.arange(frame_count, dtype=np.float32))

    def test_activation_is_steepest_positive_change(self):
        case = self.synthetic_case()
        np.testing.assert_array_equal(activation_frames(case.signals), np.arange(75, 83))

    def test_examples_are_balanced_and_labels_match_future_event(self):
        case = self.synthetic_case()
        features, labels, summary = make_examples(case, seed=7400)
        self.assertEqual(summary["skippedPoints"], 0)
        self.assertEqual(features.shape, (32, 44))
        self.assertEqual(int((labels == 0).sum()), 16)
        self.assertEqual(int((labels == 1).sum()), 16)

        for feature, label in zip(features, labels):
            point_index = int(round(feature[40] / 3))
            current_frame = int(round(feature[43] * (case.signals.shape[1] - 1)))
            activation = 75 + point_index
            expected = int(current_frame < activation <= current_frame + 20)
            self.assertEqual(int(label), expected)


if __name__ == "__main__":
    unittest.main()
