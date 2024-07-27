import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.adaptive_cruise_control import build_acc_model, train_acc_model
import os

def load_data(data_path):
    data = np.load(data_path)
    X = data['features']
    y = data['labels']
    return train_test_split(X, y, test_size=0.2)

def train(config):
    data_path = os.path.join(config['data']['processed_data_path'], 'dataset.npz')
    X_train, X_test, y_train, y_test = load_data(data_path)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_acc_model(input_shape)
    
    history = train_acc_model(model, X_train, y_train, config['training']['epochs'], config['training']['batch_size'])
    
    model.save(os.path.join(config['data']['models_path'], 'acc_model.h5'))
    return history

if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train(config)
