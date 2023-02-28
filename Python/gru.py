
# python C:\Users\Anthony\Desktop\PythonScript\x_x.py

import pandas as pd
import numpy as np
from numpy import array
import keras
from keras import backend as K
from keras.layers import Input, Dense, GRUCell, RNN, concatenate, TimeDistributed, Layer
import tensorflow as tf
from tensorflow.keras import initializers
tf.random.set_seed(69)
import os
import matplotlib.pyplot as plt
import random

def train_function(x):
    return np.sin(x)
    
def generate_train_sequences(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)+n_steps_out):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
        input_seq = array(X)
        output_seq = array(y)
    return input_seq, output_seq
    
# create sine data 
xaxis = np.arange(-50*np.pi, 50*np.pi, 0.1)
train_seq = train_function(xaxis)

# set parameters
n_steps = 20
n_features = 1
epochs = 8
batch_size = 32

# Initialize
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)
bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.15, seed=None)

def create_model(layers):  
    n_layers = len(layers)
    
    ## Encoder
    encoder_inputs = keras.layers.Input(shape=(None, 1))
    gru_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]

    encoder = keras.layers.RNN(gru_cells, return_sequences=True, return_state=True)
    encoder_outputs_and_states = encoder(encoder_inputs)
    encoder_outputs, state_h, state_h2, state_h3, state_c = encoder(encoder_inputs)
    encoder_states = encoder_outputs_and_states[1:]
    
    ## Decoder
    decoder_inputs = keras.layers.Input(shape=(None, 1))
    decoder_cells = [keras.layers.GRUCell(hidden_dim, activation='tanh', dropout=.1, recurrent_initializer=initializer, kernel_initializer=kernel_initializer,bias_initializer=bias_initializer) for hidden_dim in layers]
    decoder_gru = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    decoder_outputs_and_states = decoder_gru(decoder_inputs, initial_state=encoder_states)
    [decoder_out, forward_h, forward_h2, forward_h3, forward_c] = decoder_gru(decoder_inputs, initial_state=encoder_states)
    
    decoder_dense1 = Dense(10, activation='tanh', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs1 = decoder_dense1(decoder_out)
    decoder_dense2 = Dense(5, activation='relu', kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    decoder_outputs2 = decoder_dense2(decoder_out)
    
    merged = concatenate([decoder_outputs1, decoder_outputs2])
    
    decoder_dense = TimeDistributed(Dense(1, activation='tanh'))
    decoder_outputs = decoder_dense(merged)
    
    model = keras.models.Model([encoder_inputs,decoder_inputs], decoder_outputs)
    return model
   
neurons = 64

model = create_model([neurons,neurons,neurons,neurons])
batches = 1
def run_model(model,batches,epochs,batch_size):

    for _ in range(batches):
        input_seq, output_seq = generate_train_sequences(train_seq, n_steps, 1)
        encoder_input_data = input_seq
        decoder_target_data = output_seq
        decoder_input_data = np.zeros(decoder_target_data.shape)
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=.3,
            shuffle=False)
        total_loss.append(history.history['loss'])
        total_val_loss.append(history.history['val_loss'])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss = 'mse')
total_loss = []
total_val_loss = []

run_model(model,batches=batches, epochs=epochs,batch_size=batch_size)

def plot_loss(train_loss,val_loss):
    plt.figure(figsize=(10,6))
    plt.plot(train_loss)
    plt.plot(val_loss)

    plt.xlabel('Epoch')
    plt.ylabel('Mean Sqquared Error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train','Valid'])
    plt.show()

total_loss = [j for i in total_loss for j in i]
total_val_loss = [j for i in total_val_loss for j in i]
plot_loss(total_loss,total_val_loss)

# create test data and make predictions
test_xaxis = np.arange(0, 10*np.pi, 0.1)

def test_function(x):
    return np.sin(x)
    
seq = test_function(test_xaxis)

test_seq = seq[:n_steps]
results = []
for i in range(len(test_xaxis) - n_steps):
    input_seq_test = test_seq[i:i+n_steps].reshape((1,n_steps,1))
    decoder_input_test = np.zeros((1,1,1))
    y = model.predict([input_seq_test, decoder_input_test])
    test_seq = np.append(test_seq, y)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_true = seq[n_steps:], y_pred = test_seq[n_steps:])

plt.plot(test_xaxis[n_steps:], test_seq[n_steps:], label="predictions")
plt.plot(test_xaxis, seq, label="ground truth")
plt.plot(test_xaxis[:n_steps], test_seq[:n_steps], label="initial sequence", color="red")
plt.title('GRU Approximation of Sine Function: MSE = ' + str(round(MSE,4)))
plt.legend(loc='upper left')
plt.ylim(-2, 2)
plt.show()


