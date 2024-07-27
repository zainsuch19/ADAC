import numpy as np

def fuse_sensor_data(camera_data, lidar_data, radar_data):
    # Example: Weighted average of sensor data
    fused_data = (0.5 * camera_data + 0.3 * lidar_data + 0.2 * radar_data)
    return fused_data
