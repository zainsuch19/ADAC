import cv2
import numpy as np

def detect_lane_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    line_image = np.zeros_like(image)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
    combined = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined
