import unittest

from scripts.evaluate_occlusion import (
    Detection,
    GroundTruth,
    aggregate_recall,
    match_ground_truths,
    parse_annotations,
)


class AnnotationTests(unittest.TestCase):
    def test_filter_clip_and_raw_area_size(self):
        lines = [
            "-4,-3,20,20,1,1,0,2",
            "0,0,10,10,0,1,0,0",
            "0,0,10,10,1,0,0,0",
            "0,0,10,10,1,11,0,0",
            "200,200,5,5,1,2,0,0",
        ]
        ground_truths, excluded = parse_annotations(lines, 100, 80)
        self.assertEqual(len(ground_truths), 1)
        self.assertEqual(ground_truths[0].box, (0.0, 0.0, 16.0, 17.0))
        self.assertEqual(ground_truths[0].size, "small")
        self.assertEqual(excluded["score_nonpositive"], 1)
        self.assertEqual(excluded["class_0_ignored_region"], 1)
        self.assertEqual(excluded["class_11_others"], 1)
        self.assertEqual(excluded["invalid_after_clipping"], 1)


class MatchingTests(unittest.TestCase):
    def test_matching_is_class_aware_confidence_ordered_and_one_to_one(self):
        ground_truths = [
            GroundTruth((0, 0, 10, 10), 0, 0, 0),
            GroundTruth((20, 20, 30, 30), 0, 1, 1),
            GroundTruth((0, 0, 10, 10), 1, 2, 2),
        ]
        detections = [
            Detection((0, 0, 10, 10), 0, 0.9),
            Detection((0, 0, 10, 10), 0, 0.8),  # cannot match GT 0 twice
            Detection((0, 0, 10, 10), 1, 0.7),
            Detection((20, 20, 30, 30), 0, 0.6),
            Detection((20, 20, 30, 30), 2, 0.5),  # wrong class
        ]
        self.assertEqual(match_ground_truths(ground_truths, detections), {0, 1, 2})

    def test_strata_are_aggregated_from_one_global_match_set(self):
        ground_truths = [
            GroundTruth((0, 0, 10, 10), 0, 0, 0),
            GroundTruth((0, 0, 40, 40), 0, 1, 1),
            GroundTruth((0, 0, 100, 100), 1, 2, 2),
        ]
        metrics = aggregate_recall(ground_truths, {0, 2})
        self.assertEqual(metrics["overall"], {"match_count": 2, "total": 3, "recall": 2 / 3})
        self.assertEqual(metrics["occlusion"]["1"]["recall"], 0.0)
        self.assertEqual(metrics["size"]["small"]["recall"], 1.0)
        self.assertEqual(metrics["size"]["medium"]["recall"], 0.0)
        self.assertEqual(metrics["size"]["large"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
