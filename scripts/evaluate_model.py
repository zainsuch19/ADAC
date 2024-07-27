import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = os.path.join(config['data']['processed_data_path'], 'dataset.npz')
    data = np.load(data_path)
    X_test = data['features']
    y_test = data['labels']
    
    model_path = os.path.join(config['data']['models_path'], 'acc_model.h5')
    model = load_model(model_path)
    
    evaluate(model, X_test, y_test)
