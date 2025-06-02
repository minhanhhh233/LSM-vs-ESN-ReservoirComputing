import os
import numpy as np
import json
import torch

import torch.nn as nn
import torch.nn.functional as F

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render

from torch._tensor import Tensor
from experiments.metrics import METRICS, evaluate
from experiments.preprocessing import denormalize
from lsm.nest import RegressionDataset, convert_to_spike_times
from torch.utils.data import DataLoader
import math
import time
import nest
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import init_notebook_plotting, render
import matplotlib.pyplot as plt

from lsm.nest import create_model

def read_data(dataset_path, normalization_method, past_history_factor):
    # read normalization params
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
    x = x[:10]
    y = y[:10]
    # x_train = np.load(tmp_data_path + "x_train0.np.npy")
    # y_train = np.load(tmp_data_path + "y_train0.np.npy")
    # x_test = np.load(tmp_data_path + "x_test0.np.npy")
    # y_test = np.load(tmp_data_path + "y_test0.np.npy")

    data_shape = x.shape[0]  # Get the total number of samples
    train_size = int(data_shape * 0.8)  # Calculate the size of the training set

    # Split the data into training and testing sets
    x_train = x[:train_size]
    y_train = y[:train_size]
    x_test = x[train_size:]
    y_test = y[train_size:]

    # x_train = x_train[100:]
    # y_train = y_train[100:]
    # x_test = x_test[:10]
    # y_test = y_test[:10]

    #y_test_denorm = np.load(tmp_data_path + "y_test_denorm0.np.npy")
    y_test_denorm = np.asarray(
        [
            denormalize(y_test[i], norm_params[0], normalization_method)
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
    
    return x_train, y_train, x_test, y_test, y_test_denorm, norm_params

def train_one_epoch(epoch_index, model, train_loader, optimizer, criterion, time_step_per_sample):
        
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(train_loader):
        # Every data instance is an input + target pair
        inputs, target = data
        print("Original input: ", inputs)
        inputs = convert_to_spike_times(inputs, time_step_per_sample)
        print(type(inputs))
        print(inputs)
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = model(inputs)
        #print(outputs)
        #print("Targets: ", target)
        #print("Outputs. ", outputs)
        # Compute the loss and its gradients
        
        loss = criterion(outputs, target)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        if i % 1 == 0:
            last_loss = running_loss / (i+1) # loss per batch
            with open('LSM/results.txt','a') as file:
                file.write('\n')
                file.write('  predict: {} ------- target: {}\n'.format(outputs, target))
                file.write('  batch {} loss: {}\n'.format(i + 1, loss.item()))
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            
        #    running_loss = 0.

    return running_loss / len(train_loader)

def train(model, train_loader, epochs, optimizer, criterion, time_step_per_sample):
    #---------------- Train starts -----------------
    epoch_number = 0

    training_time_0 = time.time()
    for epoch in range(epochs):
        with open('LSM/results.txt','a') as file:
            file.write('EPOCH {}:\n'.format(epoch_number + 1))
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, model, train_loader, optimizer, criterion, time_step_per_sample)      
        with open('LSM/results-for-tune.txt','a') as file:
            file.write('LOSS train {} epoch {}\n'.format(avg_loss, epoch))
            file.write('\n') 
        epoch_number += 1
    
    training_time = time.time() - training_time_0
    with open('LSM/results-for-tune.txt','a') as file:
        file.write('Train time {}:\n'.format(training_time))
        file.write('LOSS train {}\n'.format(avg_loss))
        file.write('\n')

def evaluate(model, test_loader, norm_params, normalization_method, criterion, time_step_per_sample):
    # test_time_0 = time.time()
    # running_vloss = 0.0
    # # Set the model to evaluation mode, disabling dropout and using population
    # # statistics for batch normalization.
    # model.eval()

    # # Disable gradient computation and reduce memory consumption.
    # test_forecast = []
    
    # with torch.no_grad():
    #     for i, vdata in enumerate(test_loader):
    #         print("Index: ", i)
    #         vinputs, vtargets = vdata
    #         vinputs = convert_to_spike_times(vinputs, time_step_per_sample)
    #         voutputs = model(vinputs)
    #         # TODO
    #         voutputs = denormalize(
    #             voutputs, norm_params[0], method=normalization_method,
    #         )
            
    #         test_forecast.append(voutputs[0])
    #         vloss = criterion(voutputs, vtargets)
    #         running_vloss += vloss
    #         print("Loss: ", running_vloss)
            
    # test_forecast = np.array(test_forecast)
    # avg_vloss = running_vloss / (i + 1)

    # test_time = time.time() - test_time_0
    # with open('LSM/results-for-tune.txt','a') as file:
    #     file.write('Test time {}:\n'.format(test_time))
    # with open('LSM/results-for-tune.txt','a') as file:
    #     file.write('LOSS valid {}\n'.format(avg_vloss))
    #     file.write('\n')
    # return  {'loss': avg_vloss}

    test_time_0 = time.time()
    running_vloss = []
    model.eval()
    test_forecast = []

    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            vinputs, vtargets = vdata
            vinputs = convert_to_spike_times(vinputs, time_step_per_sample)
            voutputs = model(vinputs)
            voutputs = denormalize(voutputs, norm_params[0], method=normalization_method)
            test_forecast.append(voutputs[0])
            vloss = criterion(voutputs, vtargets)
            running_vloss.append(vloss.item())
    
    test_forecast = np.array(test_forecast)
    avg_vloss = np.mean(running_vloss)
    sem_vloss = np.std(running_vloss) / np.sqrt(len(running_vloss))

    test_time = time.time() - test_time_0
    with open('LSM/results-for-tune.txt', 'a') as file:
        file.write('Test time {}:\n'.format(test_time))
        file.write('LOSS valid: {} (SEM: {})\n\n'.format(avg_vloss, sem_vloss))
        
    return {'loss': (avg_vloss, sem_vloss)}




def main():
    def train_evaluate(parameterization):
        time_step_per_sample = parameterization['time_step_per_sample']
        
        syn_params = {key: value for key, value in parameterization.items() if key != 'time_step_per_sample'
                      and key != 'delay_mean_layer' and key != 'C_inp'}
        neu_params = None
        layer_params = {key: value for key, value in parameterization.items() if key == 'delay_mean_layer' or key == 'C_inp'}

        model = create_model(model_name, n_exc_neurons, n_inh_neurons, n_in, n_layers, time_step_per_sample, syn_params, neu_params, layer_params)

        #print("Current state: ", model.lsm_layers[-1]([nest.get("biological_time")], 2))
        #print_layer_info(model)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        train(model, train_loader,1, optimizer, criterion, time_step_per_sample)
        
        avg_loss = evaluate(model, test_loader, norm_params, normalization_method, criterion, time_step_per_sample)
        return avg_loss

    def optimize_loop():
        # Attach the trial
    
        ax_client.attach_trial(
            parameters={"time_step_per_sample": 50, "n_syn_exc": 2, "n_syn_inh": 1, "J_EE": 50, "J_EI": 250, "J_IE":-200, "J_II":-200,
                        'delay_mean_syn':10, 'tau_psc': 2.0, 'tau_fac_EE': 1, 'tau_fac_EI': 1790, 'tau_fac_IE': 376, 'tau_fac_II': 21,
                        'tau_rec_EE': 813, 'tau_rec_EI': 399, 'tau_rec_IE': 45, 'tau_rec_II': 706, 'U_EE': 0.59, 'U_EI': 0.049, 'U_IE': 0.016,
                        'U_II': 0.25, "delay_mean_layer":10, "C_inp": 50}
        )

        # Get the parameters and run the trial 
        baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
        ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))

        for i in range(50):
            parameters, trial_index = ax_client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters)) 

    def setup_experiment():
        # Create an experiment with required arguments: name, parameters, and objective_name.
        ax_client.create_experiment(
            name="tune_lsm-iaf_psc_exp-tsodyks",  # The name of the experiment.
            parameters=[
                {
                    "name": "time_step_per_sample",  # The name of the parameter.
                    "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                    "bounds": [10, 100],  # The bounds for range parameters. 
                    # "values" The possible values for choice parameters .
                    # "value" The fixed value for fixed parameters.
                    "value_type": "int"   
                },
                {
                    "name": "n_syn_exc",  
                    "type": "range",  
                    "bounds": [1, 5],
                    "value_type": "int"  
                },
                {
                    "name": "n_syn_inh",  
                    "type": "range",  
                    "bounds": [1, 5],
                    "value_type": "int"  
                },
                {
                    "name": "J_EE",  
                    "type": "range",  
                    "bounds": [100, 500],
                    "value_type": "int" 
                },
                {
                    "name": "J_EI",  
                    "type": "range",  
                    "bounds": [10, 500],
                    "value_type": "int"  
                },
                {
                    "name": "J_IE",  
                    "type": "range",  
                    "bounds": [-500, -10],
                    "value_type": "int"  
                },
                {
                    "name": "J_II",  
                    "type": "range",  
                    "bounds": [-500, -10],
                    "value_type": "int"  
                },
                {
                    "name": "delay_mean_syn",  
                    "type": "range",  
                    "bounds": [1, 50],
                    "value_type": "int" 
                },
                {
                    "name": "tau_psc",  
                    "type": "range",  
                    "bounds": [0.1, 5],
                    "value_type": "float" 
                },
                {
                    "name": "tau_fac_EE",  
                    "type": "range",  
                    "bounds": [1, 5000],
                    "value_type": "int" 
                },
                {
                    "name": "tau_fac_EI",  
                    "type": "range",  
                    "bounds": [1, 5000],
                    "value_type": "int"  
                },
                {
                    "name": "tau_fac_IE",  
                    "type": "range",  
                    "bounds": [1, 5000],
                    "value_type": "int"  
                },
                {
                    "name": "tau_fac_II",  
                    "type": "range",  
                    "bounds": [1, 5000],
                    "value_type": "int"  
                },
                {
                    "name": "tau_rec_EE",  
                    "type": "range",  
                    "bounds": [1, 1000],
                    "value_type": "int" 
                },
                {
                    "name": "tau_rec_EI",  
                    "type": "range",  
                    "bounds": [1, 1000],
                    "value_type": "int"  
                },
                {
                    "name": "tau_rec_IE",  
                    "type": "range",  
                    "bounds": [1, 1000],
                    "value_type": "int"  
                },
                {
                    "name": "tau_rec_II",  
                    "type": "range",  
                    "bounds": [1, 1000],
                    "value_type": "int"  
                },
                {
                    "name": "U_EE",  
                    "type": "range",  
                    "bounds": [0, 1],
                    "value_type": "float" 
                },
                {
                    "name": "U_EI",  
                    "type": "range",  
                    "bounds": [0, 1],
                    "value_type": "float"  
                },
                {
                    "name": "U_IE",  
                    "type": "range",  
                    "bounds": [0, 1],
                    "value_type": "float"  
                },
                {
                    "name": "U_II",  
                    "type": "range",  
                    "bounds": [0, 1],
                    "value_type": "float"  
                },
                {
                    "name": "delay_mean_layer",  
                    "type": "range",  
                    "bounds": [1, 50],
                    "value_type": "int"  
                },
                {
                    "name": "C_inp",  
                    "type": "range",  
                    "bounds": [1, 108],
                    "value_type": "int"  
                }
            ],
            objectives={"loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
            # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
            # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
        )  

    init_notebook_plotting()

    torch.manual_seed(42)
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = 'data/ExchangeRate'
    normalization_method = 'minmax'
    past_history_factor = 1.25
    batch_size = 1
    learning_rate = 0.001

    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor
    )

    train_dataset = RegressionDataset(x_train, y_train)
    test_dataset = RegressionDataset(x_test, y_test_denorm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)

    forecast_horizon = y_test.shape[1]
    past_history = x_test.shape[1]
    # steps_per_epoch = min(
    #     int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    # )

    model_name = 'lsm-iaf_psc_exp-tsodyks'
    n_total_neurons = 32
    n_exc_neurons = int(math.ceil(n_total_neurons * 0.8))
    n_inh_neurons = n_total_neurons - n_exc_neurons
    #n_rec_neurons = 50
    n_in = 8
    n_layers = 1

    ax_client = AxClient()
    setup_experiment()
    optimize_loop()
    best_parameters, values = ax_client.get_best_parameters()
    print("Best_parameter: ", best_parameters)
    mean, covariance = values
    print("Mean: ", mean, " covariance: ", covariance)
    
    render(ax_client.get_contour_plot(param_x="time_step_per_sample", param_y="n_syn_exc", metric_name="loss")) 
    

    
    


if __name__ == "__main__":
    main()