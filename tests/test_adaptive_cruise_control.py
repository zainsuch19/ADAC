# tests/test_adaptive_cruise_control.py
import unittest
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.adaptive_cruise_control import build_acc_model, train_acc_model

class TestAdaptiveCruiseControl(unittest.TestCase):
    def setUp(self):
        self.input_shape = (10, 5)
        self.train_data = np.random.random((100, 10, 5))
        self.train_labels = np.random.random((100, 1))
        self.epochs = 1
        self.batch_size = 10
    
    def test_build_acc_model(self):
        model = build_acc_model(self.input_shape)
        self.assertIsInstance(model, Sequential, "Model is not an instance of Sequential")
        self.assertEqual(len(model.layers), 3, "Model does not have 3 layers")
        self.assertIsInstance(model.layers[0], LSTM, "First layer is not an LSTM layer")
        self.assertIsInstance(model.layers[2], Dense, "Last layer is not a Dense layer")
    
    def test_train_acc_model(self):
        model = build_acc_model(self.input_shape)
        history = train_acc_model(model, self.train_data, self.train_labels, self.epochs, self.batch_size)
        self.assertIsNotNone(history, "Training history is None")
        self.assertIn('loss', history.history, "Training history does not contain 'loss'")

if __name__ == "__main__":
    unittest.main()
