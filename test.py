import sys
import os
import cv2

# Add the ADAC Project root directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the function from lane_departure_warning.py
from src.lane_departure_warning import detect_lane_lines

def test_lane_departure():
    # Use raw strings to avoid unicode errors in the file path
    image_path = r"C:\Users\zaina\Documents\ADAC Project\Screenshot 2024-06-24 172905.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image at {image_path}")
        return
    
    lane_image = detect_lane_lines(image)
    
    cv2.imshow("Lane Detection", lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_lane_departure()
