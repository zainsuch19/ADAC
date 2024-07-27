import unittest
import cv2
import os
import sys

# Add the src directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.lane_departure_warning import detect_lane_lines

class TestLaneDepartureWarning(unittest.TestCase):
    def setUp(self):
        self.image_path = r"C:\Users\zaina\Documents\ADAC Project\Screenshot 2024-06-24 171345.png"
        self.image = cv2.imread(self.image_path)
    
    def test_detect_lane_lines(self):
        self.assertIsNotNone(self.image, "Failed to load test image")
        lane_image = detect_lane_lines(self.image)
        self.assertIsNotNone(lane_image, "Lane detection failed")
        self.assertEqual(self.image.shape, lane_image.shape, "Output image shape mismatch")

if __name__ == "__main__":
    unittest.main()
