import os
import argparse
import json
import itertools
import time
import requests
from multiprocessing import Process, Manager
import numpy as np
import pandas as pd
from experiments.metrics import METRICS, evaluate
from experiments.preprocessing import denormalize
import matplotlib.pyplot as plt

def read_data(dataset_path, normalization_method, past_history_factor, n_train_sample):
    # print("Number of train zise", n_train_sample, type(n_train_sample))
    # read normalization params
    n_train_sample = int(n_train_sample)
    norm_params = None
    with open(
        os.path.normpath(dataset_path)
        + "/{}/norm_params.json".format(normalization_method),
        "r",
    ) as read_file:
        norm_params = json.load(read_file)

    # read training / validation data
    tmp_data_path = os.path.normpath(dataset_path) + "/{}/{}/".format(
        normalization_method, past_history_factor
    )
    x = np.load(tmp_data_path + "x_train0.np.npy")
    y = np.load(tmp_data_path + "y_train0.np.npy")
    #x = x[:1250]
    #y = y[:1250]
    norm_params_index = 0

    # x_train = np.load(tmp_data_path + "x_train0.np.npy")
    # y_train = np.load(tmp_data_path + "y_train0.np.npy")
    # x_test = np.load(tmp_data_path + "x_test0.np.npy")
    # y_test = np.load(tmp_data_path + "y_test0.np.npy")

    data_shape = x.shape[0]  # Get the total number of samples
    train_size = int(data_shape * 0.8)  # Calculate the size of the training set
    
    # Split the data into training and testing sets
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_train = x[:n_train_sample]
    y_train = y[:n_train_sample]
    x_test = x[train_size:]
    y_test = y[train_size:]

    # x_train = x_train[-100:]
    # y_train = y_train[-100:]
    # x_test = x_test[:10]
    # y_test = y_test[:10]

    #y_test_denorm = np.load(tmp_data_path + "y_test_denorm0.np.npy")
    y_test_denorm = np.asarray(
        [
            denormalize(y_test[i], norm_params[norm_params_index], normalization_method)
            for i in range(y_test.shape[0])
        ]
    )
    
    #x_train = x_train[:10]
    #y_train = y_train[:10]
    x_train = np.squeeze(x_train) # To convert timesteps to number of input neurons
    x_test = x_test[:, :, 0] # To convert timesteps to number of input neurons

    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)
    
    return x_train, y_train, x_test, y_test, y_test_denorm, norm_params[norm_params_index]

def main():
    dataset_path = "data/ExchangeRate"
    normalization_method = "minmax"
    past_history_factor = 1.25
    n_train_sample = 1000
    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor, n_train_sample
    )
    y_predict = np.load("testLsm/ExchangeRate/minmax/1.25/1/1/0.001/lsm-iaf_psc_exp-tsodyks/32/1/1704645637/0.npy")
    print(y_predict.shape)
    y_predict = y_predict[:, 0].reshape(-1, 1)
    y_train = y_train[:, 0].reshape(-1, 1)
    y_test_denorm = y_test_denorm[:, 0].reshape(-1, 1)
    print(y_predict.shape)
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

    # Plotting y_predict
    plt.plot(y_predict, label='Predicted Data', color='blue')

    # Plotting y_test_denorm
    plt.plot(y_test_denorm, label='True Data (Denormalized)', color='green')

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Comparison between Predicted and True Denormalized Data')
    plt.legend()  # Displaying legend

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

