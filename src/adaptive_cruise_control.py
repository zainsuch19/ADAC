import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore

def build_acc_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    return model

def train_acc_model(model, train_data, train_labels, epochs, batch_size):
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
    return history
