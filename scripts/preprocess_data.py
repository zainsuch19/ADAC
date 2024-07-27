import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_images(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        processed_image = cv2.resize(image, (224, 224))
        cv2.imwrite(os.path.join(output_dir, filename), processed_image)

def preprocess_data(config):
    image_dir = config['data']['raw_data_path']
    output_dir = config['data']['processed_data_path']
    preprocess_images(image_dir, output_dir)

if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    preprocess_data(config)
