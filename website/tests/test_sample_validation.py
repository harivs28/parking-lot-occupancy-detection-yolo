import json
import unittest
from pathlib import Path

import cv2

from website.app import DetectionSettings, process_frame


ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = ROOT_DIR / "results" / "FULL_IMAGE_1000x750"
EXPECTATIONS_PATH = ROOT_DIR / "results" / "sample_image_expectations.json"


class SampleImageValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expectations = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
        cls.settings = DetectionSettings()

    def test_bundled_sample_images_match_expected_counts(self):
        for sample in self.expectations["samples"]:
            with self.subTest(image=sample["image"]):
                image_path = SAMPLE_ROOT / sample["image"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image, f"Could not read sample image: {image_path}")

                result = process_frame(image, settings=self.settings)
                stats = result.stats

                self.assertEqual(stats["camera_label"], sample["expected_camera_label"])
                self.assertEqual(stats["slot_mode"], sample["expected_slot_mode"])
                self.assertEqual(stats["total"], sample["expected_total"])
                self.assertEqual(stats["empty"], sample["expected_empty"])
                self.assertEqual(stats["occupied"], sample["expected_occupied"])


if __name__ == "__main__":
    unittest.main()
