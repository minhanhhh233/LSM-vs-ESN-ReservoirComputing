import sys
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
from lsm.nest import RegressionDataset, convert_to_spike_times_latency, convert_to_spike_times, convert_to_spike_times_delta
from torch.utils.data import DataLoader
import math
import torch

def notify_slack(msg, webhook=None):
    if webhook is None:
        webhook = os.environ.get("webhook_slack")
    if webhook is not None:
        try:
            requests.post(webhook, json.dumps({"text": msg}))
        except:
            print("Error while notifying slack")
            print(msg)
    else:
        print("NO WEBHOOK FOUND")


def check_params(datasets, models, results_path, parameters, metrics, csv_filename):
    assert len(datasets) > 0, "dataset parameter is not well defined."
    assert all(
        os.path.exists(ds_path) for ds_path in datasets
    ), "dataset paths are not well defined."
    assert all(
        param in parameters.keys()
        for param in [
            "normalization_method",
            "past_history_factor",
            "batch_size",
            "epochs",
            "max_steps_per_epoch",
            "learning_rate",
            "model_params",
        ]
    ), "Some parameters are missing in the parameters file."
    assert all(
        model in parameters["model_params"] for model in models
    ), "models parameter is not well defined."
    assert metrics is None or all(m in METRICS.keys() for m in metrics)


def read_results_file(csv_filepath, metrics):
    # try:
    #     results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    # except IOError:
    #     results = pd.DataFrame(
    #         columns=[
    #             "DATASET",
    #             "MODEL",
    #             "MODEL_INDEX",
    #             "MODEL_DESCRIPTION",
    #             "FORECAST_HORIZON",
    #             "PAST_HISTORY_FACTOR",
    #             "PAST_HISTORY",
    #             "BATCH_SIZE",
    #             "EPOCHS",
    #             "STEPS",
    #             "OPTIMIZER",
    #             "LEARNING_RATE",
    #             "NORMALIZATION",
    #             "TEST_TIME",
    #             "TRAINING_TIME",
    #             *metrics,
    #             "LOSS",
    #             "VAL_LOSS",
    #         ]
    #     )
    results = pd.DataFrame(
        columns=[
            "DATASET",
            "TRAINING_SIZE",
            "MODEL",
            "MODEL_INDEX",
            "MODEL_DESCRIPTION",
            "FORECAST_HORIZON",
            "PAST_HISTORY_FACTOR",
            "PAST_HISTORY",
            "BATCH_SIZE",
            "EPOCHS",
            "STEPS",
            "OPTIMIZER",
            "LEARNING_RATE",
            "NORMALIZATION",
            "TEST_TIME",
            "TRAINING_TIME",
            *metrics,
            "LOSS",
            "STD_LOSS",
            "VAL_LOSS",
            "STD_VAL_LOSS"
        ]
    )
    return results


def product(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


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
    x = x[:-696]
    y = y[:-696]

    #x = x[:1250]
    #y = y[:1250]
    norm_params_index = 0

    # x_train = np.load(tmp_data_path + "x_train0.np.npy")
    # y_train = np.load(tmp_data_path + "y_train0.np.npy")
    # x_test = np.load(tmp_data_path + "x_test0.np.npy")
    # y_test = np.load(tmp_data_path + "y_test0.np.npy")

    data_shape = x.shape[0]  # Get the total number of samples
    # train_size = int(data_shape * 0.8)  # Calculate the size of the training set
    train_size = 6183
    
    # Split the data into training and testing sets
    #x_train = x[:train_size]
    #y_train = y[:train_size]

    """
    # 10th fold
    
    x_train = x[:n_train_sample]
    y_train = y[:n_train_sample]
    x_test = x[train_size:]
    y_test = y[train_size:]
    """
    

    """
    # 1st fold
    x_test = x[:687]
    y_test = y[:687]

    x = x[687:]
    y = y[687:]
    x_train = x[:n_train_sample]
    y_train = y[:n_train_sample]
    """
    

    """
    # 2nd fold
    x_train = x[:687]
    y_train = y[:687]
    x_test = x[687:(687*2)]
    y_test = y[687:(687*2)]
    
    x_train = np.concatenate((x_train, x[(687*2):]), axis=0)
    y_train = np.concatenate((y_train, y[(687*2):]), axis=0)

    x_train = x_train[:n_train_sample]
    y_train = y_train[:n_train_sample]
    """


    
    # nth fold
    x_train = x[:(687*2)]
    y_train = y[:(687*2)]
    x_test = x[(687*2):(687*3)]
    y_test = y[(687*2):(687*3)]
    
    x_train = np.concatenate((x_train, x[(687*3):]), axis=0)
    y_train = np.concatenate((y_train, y[(687*3):]), axis=0)
    
    x_train = x_train[:n_train_sample]
    y_train = y_train[:n_train_sample]
    
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

def train_one_epoch(epoch_index, model, train_loader, optimizer, criterion, time_step_per_sample):
        
    running_loss = []
    last_loss = 0.
    # all_outputs = []
    for i, data in enumerate(train_loader):
        # Every data instance is an input + target pair
        inputs, target = data
        inputs = convert_to_spike_times_latency(inputs, time_step_per_sample, 1)
        # print("Encoded Inputs: ", inputs)
        
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
        running_loss.append(loss.item())
        # all_outputs.append(outputs.detach().cpu().numpy())
        
        # if i % 1 == 0:
        #     last_loss = running_loss / (i+1) # loss per batch
        #     with open('LSM/results.txt','a') as file:
        #         file.write('\n')
        #         file.write('  predict: {} ------- target: {}\n'.format(outputs, target))
        #         file.write('  batch {} loss: {}\n'.format(i + 1, loss.item()))
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(train_loader) + i + 1
            
        #    running_loss = 0.
    
    avg_loss = np.mean(running_loss)
    std_loss = np.std(running_loss)

    """
    with open('LSM/results.txt', 'a') as file:
        file.write('LOSS epoch {}: {} (STD: {})\n\n'.format(epoch_index, avg_loss, std_loss))
    """
    # outputs_array = np.concatenate(all_outputs, axis=0)  # Combine all outputs into one array
    
    # Define the path to save the outputs
    # outputs_path = 'LSM/testLsm/outputs_epoch_{}.npy'.format(epoch_index)
    
    # Save all outputs to a NumPy file
    # np.save(outputs_path, outputs_array)
        
    return {'loss': (avg_loss, std_loss)}

def train(model, train_loader, epochs, optimizer, criterion, time_step_per_sample):
    #---------------- Train starts -----------------
    epoch_number = 0
    
    
    training_time_0 = time.time()
    # overall_avg_loss = []
    # overall_std = []
    """
    training_results = pd.DataFrame(
        columns=[
            "EPOCH",
            "LOSS",
            "STD_LOSS"
        ]
    )
    """

    for epoch in range(epochs):
        # with open('LSM/results.txt','a') as file:
        #     file.write('EPOCH {}:\n'.format(epoch_number + 1))
        # print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        result = train_one_epoch(epoch_number, model, train_loader, optimizer, criterion, time_step_per_sample)  
        avg_loss_1_epoch, std_loss_1_epoch = result['loss']
        # overall_avg_loss.append(avg_loss_1_epoch)
        # overall_std.append(std_loss_1_epoch)
        # with open('LSM/results-for-tune.txt','a') as file:
        #     file.write('LOSS train {} epoch {}\n'.format(avg_loss, epoch))
        #     file.write('\n')
        """
        training_results = training_results.append(
            {
                "EPOCH": epoch_number,
                "LOSS": avg_loss_1_epoch,
                "STD_LOSS":std_loss_1_epoch
            },
            ignore_index=True,
        )
        
        training_results.to_csv(
            "LSM/testLsm/training_results.csv", sep=";", mode='a', index=True, header=True
        )
        """
        epoch_number += 1

    
    training_time = time.time() - training_time_0
    # avg_loss = np.mean(overall_avg_loss)
   
    # std_loss = np.sqrt(
    #     (np.sum(np.square(overall_std)) + 
    #     np.sum(np.square(overall_avg_loss - avg_loss))) / len(overall_std)
    # )
    # print("Model ID: ", id(model))
    with open('LSM/results.txt','a') as file:
        file.write('Train time {}:\n'.format(training_time))
        file.write('LOSS train: {} (STD: {})\n'.format(avg_loss_1_epoch, std_loss_1_epoch))
        file.write('\n')
    return {'loss': (avg_loss_1_epoch, std_loss_1_epoch)}, training_time


def evaluate_main(model, test_loader, y_test_denorm, metrics, norm_params, normalization_method, criterion, time_step_per_sample):

    test_time_0 = time.time()
    running_vloss = []
    model.eval()
    test_forecast = []
    
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            vinputs, vtargets = vdata
            vinputs = convert_to_spike_times_latency(vinputs, time_step_per_sample, 1)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vtargets)
            running_vloss.append(vloss.item())
            # print("Evaluation output:", voutputs)
            # print("Evaluation target: ", vtargets)

            voutputs_denorm = denormalize(voutputs, norm_params, method=normalization_method)
            test_forecast.append(voutputs_denorm[0])
    
    test_forecast = np.array(test_forecast)
    avg_vloss = np.mean(running_vloss)
    std_vloss = np.std(running_vloss)

    test_time = time.time() - test_time_0
    
    with open('LSM/results.txt', 'a') as file:
        file.write('Test time {}:\n'.format(test_time))
        file.write('LOSS valid: {} (STD: {})\n\n'.format(avg_vloss, std_vloss))
    
    print("metrics starts")
    # print("Validation predict: ", test_forecast)
    # print("Validation target; ", y_test_denorm)
    # print("Y test predict: ", test_forecast)
    # print("Y test denorm: ", y_test_denorm)
    if metrics:
        test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
    else:
        test_metrics = {}
    # print("Model ID: ", id(model))
    return {'loss': (avg_vloss, std_vloss)}, test_time, test_metrics, test_forecast

def _run_experiment(
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
    n_train_sample
):
    
    import gc
    # import tensorflow as tf
    # from models import create_model


    # tf.keras.backend.clear_session()

    # def select_gpu_device(gpu_number):
    #     gpus = tf.config.experimental.list_physical_devices("GPU")
    #     if len(gpus) >= 2 and gpu_number is not None:
    #         device = gpus[gpu_number]
    #         tf.config.experimental.set_memory_growth(device, True)
    #         tf.config.experimental.set_visible_devices(device, "GPU")

    # select_gpu_device(gpu_device)

    results = read_results_file(csv_filepath, metrics)
    from lsm.nest import create_model
    
    
    from lsm.nest.utils import print_layer_info
    import nest

    nest.ResetKernel()
    nest.rng_seed = 1
    np.random.seed(0)
    
    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor, n_train_sample
    )
    
    train_dataset = RegressionDataset(x_train, y_train)
    test_dataset = RegressionDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset)
    
    forecast_horizon = y_test.shape[1]
    past_history = x_test.shape[1]
    steps_per_epoch = min(
        int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    )

    # Best one here: Stop changing: Mean:  {'loss': 0.00020278171513949186}  covariance:  {'loss': {'loss': 1.4169265486344957e-09}}
    #parameterization = {'time_step_per_sample': 79, 'n_syn_exc': 3, 'n_syn_inh': 4, 'J_EE': 287, 'J_EI': 233, 'J_IE': -350, 'J_II': -102, 'delay_mean_syn': 2, 'tau_psc': 3.9519516647894792, 'tau_fac_EE': 2161, 'tau_fac_EI': 2185, 'tau_fac_IE': 1072, 'tau_fac_II': 324, 'tau_rec_EE': 848, 'tau_rec_EI': 537, 'tau_rec_IE': 317, 'tau_rec_II': 514, 'U_EE': 0.6670510719909651, 'U_EI': 0.5362921114875793, 'U_IE': 0.4983123922715824, 'U_II': 0.7316190591241678, 'delay_mean_layer': 19, 'learning_rate': 0.04898399260292055}

    #parameterization = {'time_step_per_sample': 84, 'n_syn_exc': 2, 'n_syn_inh': 5, 'J_EE': 374, 'J_EI': 371, 'J_IE': -313, 'J_II': -451, 'delay_mean_syn': 6, 'tau_psc': 2.0638866311807185, 'tau_fac_EE': 801, 'tau_fac_EI': 960, 'tau_fac_IE': 1366, 'tau_fac_II': 3206, 'tau_rec_EE': 179, 'tau_rec_EI': 490, 'tau_rec_IE': 708, 'tau_rec_II': 427, 'U_EE': 0.26275524912947124, 'U_EI': 0.3182624303677595, 'U_IE': 0.6850234735659967, 'U_II': 0.7999439474589006, 'delay_mean_layer': 50}
    parameterization = {'time_step_per_sample': 50, 'n_syn_exc': 2, 'n_syn_inh': 1, 'J_EE': 50, 'J_EI': 250, 'J_IE': -200, 'J_II': -200, 'delay_mean_syn': 10, 'delay_mean_layer': 10}
    
    #parameterization = {'time_step_per_sample': 100, 'n_syn_exc': 3, 'n_syn_inh': 2, 'J_EE': 231, 'J_EI': 255, 'J_IE': -340, 'J_II': -67, 'delay_mean_syn': 2, 'tau_psc': 1.3222192828921695, 'tau_fac_EE': 2720, 'tau_fac_EI': 2943, 'tau_fac_IE': 4003, 'tau_fac_II': 2560, 'tau_rec_EE': 383, 'tau_rec_EI': 332, 'tau_rec_IE': 176, 'tau_rec_II': 261, 'U_EE': 0.3980825719908307, 'U_EI': 0.3850927738429605, 'U_IE': 0.7041786564544147, 'U_II': 0.5739409524323823, 'delay_mean_layer': 43, 'C_inp': 70}
    # parameterization = {'time_step_per_sample': 233, 'n_syn_exc': 20, 'n_syn_inh': 4, 'J_EE': 470, 'J_EI': 363, 'J_IE': -72, 'J_II': -211, 'delay_mean_syn': 33, 'tau_psc': 2.0495311947539445, 'tau_fac_EE': 3769, 'tau_fac_EI': 4926, 'tau_fac_IE': 1320, 'tau_fac_II': 4716, 'tau_rec_EE': 939, 'tau_rec_EI': 27, 'tau_rec_IE': 921, 'tau_rec_II': 210, 'U_EE': 0.6956528425216675, 'U_EI': 0.1582450708374381, 'U_IE': 0.2571993609890342, 'U_II': 0.03181229345500469, 'delay_mean_layer': 15, 'C_inp': 94}
    # Mean:  {'loss': 0.0046365976070512314}  covariance:  {'loss': {'loss': 1.1351198540580064e-08}}
    # parameterization = {'time_step_per_sample': 194, 'n_rec_neurons': 44, 'n_syn_exc': 7, 'n_syn_inh': 3, 'J_EE': 118, 'J_EI': 317, 'J_IE': -458, 'J_II': -27, 'delay_mean_syn': 48, 'tau_psc': 1.9501301262527706, 'tau_fac_EE': 3871, 'tau_fac_EI': 4465, 'tau_fac_IE': 3181, 'tau_fac_II': 810, 'tau_rec_EE': 133, 'tau_rec_EI': 242, 'tau_rec_IE': 95, 'tau_rec_II': 855, 'U_EE': 0.9601608356460929, 'U_EI': 0.9252196941524744, 'U_IE': 0.7931821569800377, 'U_II': 0.09413426462560892, 'delay_mean_layer': 1, 'C_inp': 38}
    
    # parameterization = {'time_step_per_sample': 42, 'n_rec_neurons': 16, 'n_syn_exc': 49, 'n_syn_inh': 17, 'J_EE': 39, 'J_EI': 311, 'J_IE': -93, 'J_II': -332,
    #                      'delay_mean_syn': 31, 'tau_psc': 2.243894429598004, 'tau_fac_EE': 2336, 'tau_fac_EI': 4097, 'tau_fac_IE': 3398, 'tau_fac_II': 1642, 
    #                      'tau_rec_EE': 949, 'tau_rec_EI': 723, 'tau_rec_IE': 185, 'tau_rec_II': 832, 'U_EE': 0.7582208132371306, 'U_EI': 0.026171814650297165, 
    #                      'U_IE': 0.41683235112577677, 'U_II': 0.24518325459212065, 'delay_mean_layer': 40, 'C_inp': 105}
    # parameterization = {'time_step_per_sample': 68, 'n_rec_neuron': 50,'n_syn_exc': 8, 'n_syn_inh': 6, 'J_EE': 301, 'J_EI': 322, 'J_IE': -143, 'J_II': -251,
    #                      'delay_mean_syn': 9, 'tau_psc': 2.107828327082098, 'tau_fac_EE': 4681, 'tau_fac_EI': 3176, 'tau_fac_IE': 540,
    #                      'tau_fac_II': 1573, 'tau_rec_EE': 392, 'tau_rec_EI': 768, 'tau_rec_IE': 500, 'tau_rec_II': 835, 'U_EE': 0.045857843942940235,
    #                      'U_EI': 0.6986421272158623, 'U_IE': 0.33333680499345064, 'U_II': 0.0454063955694437, 'delay_mean_layer': 18, 'C_inp': 27}
    # parameterization = {'time_step_per_sample': 48, 'n_syn_exc': 49, 'n_syn_inh': 4, 'J_EE': 14, 'J_EI': 83, 'J_IE': -71, 'J_II': -209,
    #                     'delay_mean_syn': 30, 'tau_psc': 4.507326194643975, 'tau_fac_EE': 86, 'tau_fac_EI': 566, 'tau_fac_IE': 85, 'tau_fac_II': 4529,
    #                     'tau_rec_EE': 510, 'tau_rec_EI': 75, 'tau_rec_IE': 35, 'tau_rec_II': 237, 'U_EE': 0.7403305182233453,
    #                     'U_EI': 0.4271291811019182, 'U_IE': 0.6274703629314899, 'U_II': 0.3040588106960058, 'delay_mean_layer': 30, 'C_inp': 87}

    time_step_per_sample = parameterization['time_step_per_sample']
    # time_step_per_sample = 1
    
    syn_params = {key: value for key, value in parameterization.items() if key != 'time_step_per_sample' and
                    key != 'delay_mean_layer'}
    neu_params = None
    layer_params = {key: value for key, value in parameterization.items() if key == 'delay_mean_layer'}

    n_total_neurons = model_args['units']
    n_exc_neurons = int(math.ceil(n_total_neurons * 0.8))
    n_inh_neurons = n_total_neurons - n_exc_neurons
    #n_rec_neurons = min(50, n_exc_neurons)
    # TODO: Rememner to fix this when change data set
    n_in = 1
    n_layers = model_args['layers']

    model = create_model(model_name, n_exc_neurons, n_inh_neurons, n_in, n_layers, time_step_per_sample, syn_params, neu_params, layer_params)
    # print("Model ID: ", id(model))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_result, training_time = train(model, train_loader, epochs, optimizer, criterion, time_step_per_sample)
    
    avg_loss, sem_loss = train_result['loss']
    print("------- Finish training -------")
    print("------- Start testing -------")
    # print(id(model))
    test_result, test_time, test_metrics, test_forecast = evaluate_main(model, test_loader, y_test_denorm, metrics, norm_params, normalization_method, criterion, time_step_per_sample)
    avg_vloss, sem_vloss = test_result['loss']
    print("------- Finish testing -------")
    # n_total_neurons = model_args['units']
    # n_exc_neurons = int(math.ceil(n_total_neurons * 0.8))
    # n_inh_neurons = n_total_neurons - n_exc_neurons
    # n_rec_neurons = 50

    # n_layers = model_args['layers']

    # best_vloss = 1_000_000.
    # epoch_number = 0
    # print("At least here")
    # #train_loader, test_loader = read_data(batch_size)
    
    # model = create_model(model_name, n_exc_neurons, n_inh_neurons, n_rec_neurons, n_layers)
    
    # print("Current state: ", model.lsm_layers[-1]([nest.get("biological_time")], 2))
    # #print_layer_info(model)
    
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # #---------------- Train starts -----------------
    # training_time_0 = time.time()
    # for epoch in range(epochs):
    #     with open('LSM/results.txt','a') as file:
    #         file.write('EPOCH {}:\n'.format(epoch_number + 1))
    #     print('EPOCH {}:'.format(epoch_number + 1))

    #     # Make sure gradient tracking is on, and do a pass over the data
    #     model.train(True)
    #     avg_loss = train_one_epoch(epoch_number, model, train_loader, optimizer, criterion)      
    #     with open('LSM/results.txt','a') as file:
    #         file.write('LOSS train {} epoch {}\n'.format(avg_loss, epoch))
    #         file.write('\n') 
    #     epoch_number += 1

    # training_time = time.time() - training_time_0
    # with open('LSM/results.txt','a') as file:
    #     file.write('Train time {}:\n'.format(training_time))
    
    # # --------------------- Finish training ------------------------
    
    # # --------------------- Start testing ------------------------
    # print("test starts")
    
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
    #         vinputs = convert_to_spike_times(vinputs)
    #         voutputs = model(vinputs)
    #         # TODO
    #         voutputs = denormalize(
    #             voutputs, norm_params, method=normalization_method,
    #         )
            
    #         test_forecast.append(voutputs[0])
    #         vloss = criterion(voutputs, vtargets)
    #         running_vloss += vloss
    #         print("Loss: ", running_vloss)
            
    # test_forecast = np.array(test_forecast)
    
    
    # avg_vloss = running_vloss / (i + 1)

    # test_time = time.time() - test_time_0
    # with open('LSM/results.txt','a') as file:
    #     file.write('Test time {}:\n'.format(test_time))
    # with open('LSM/results.txt','a') as file:
    #     file.write('LOSS train {} valid {}\n'.format(avg_loss, avg_vloss))
    #     file.write('\n')
    
    # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    
    # print("metrics starts")
    # print("Validation predict: ", test_forecast)
    # print("Validation target; ", y_test_denorm)
    # if metrics:
    #     test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
    # else:
    #     test_metrics = {}

    
    time_stamp = int(time.time())
    # Save results
    predictions_path = "{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/".format(
        results_path,
        dataset,
        normalization_method,
        past_history_factor,
        epochs,
        batch_size,
        learning_rate,
        model_name,
        n_total_neurons,
        n_layers,
        len(x_train),
        time_stamp
    )
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    np.save(
        predictions_path + str(model_index) + ".npy", test_forecast,
    )
    
    results = results.append(
        {
            "DATASET": dataset,
            "TRAINING_SIZE": len(x_train),
            "MODEL": model_name,
            "MODEL_INDEX": id(model),
            "MODEL_DESCRIPTION": str(model_args),
            "FORECAST_HORIZON": forecast_horizon,
            "PAST_HISTORY_FACTOR": past_history_factor,
            "PAST_HISTORY": past_history,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "STEPS": steps_per_epoch,
            "OPTIMIZER": "Adam",
            "LEARNING_RATE": learning_rate,
            "NORMALIZATION": normalization_method,
            "TEST_TIME": test_time,
            "TRAINING_TIME": training_time,
            **test_metrics,
            "LOSS": avg_loss,
            "STD_LOSS":sem_loss,
            "VAL_LOSS": avg_vloss,
            "STD_VAL_LOSS": sem_vloss
        },
        ignore_index=True,
    )
    
    results.to_csv(
        csv_filepath, sep=";", mode='a', index=True, header=True
    )
    
    model = model.cpu()
    del model, x_train, x_test, y_train, y_test, y_test_denorm, test_forecast
    gc.collect()
    
    torch.cuda.empty_cache() 


def run_experiment(
    error_dict,
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
    n_train_sample
):

    try:    
        _run_experiment(
            gpu_device,
            dataset,
            dataset_path,
            results_path,
            csv_filepath,
            metrics,
            epochs,
            normalization_method,
            past_history_factor,
            max_steps_per_epoch,
            batch_size,
            learning_rate,
            model_name,
            model_index,
            model_args,
            n_train_sample
        )
    except Exception as e:
        error_dict["status"] = -1
        error_dict["message"] = str(e)
    else:
        error_dict["status"] = 1


def main(args):
    datasets = args.datasets
    models = args.models
    n_train_sample = args.n_train_sample
    results_path = args.output
    gpu_device = args.gpu
    metrics = args.metrics
    csv_filename = args.csv_filename
    epochss = args.epochss

    parameters = None
    with open(args.parameters, "r") as params_file:
        parameters = json.load(params_file)

    check_params(datasets, models, results_path, parameters, metrics, csv_filename)

    if len(models) == 0:
        models = list(parameters["model_params"].keys())

    if metrics is None:
        metrics = list(METRICS.keys())

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for dataset_index, dataset_path in enumerate(datasets):
        dataset = os.path.basename(os.path.normpath(dataset_path))

        csv_filepath = results_path + "/{}/{}".format(dataset, csv_filename)
        results = read_results_file(csv_filepath, metrics)
        current_index = results.shape[0]
        print("CURRENT INDEX", current_index)

        experiments_index = 0
        num_total_experiments = np.prod(
            [len(parameters[k]) for k in parameters.keys() if k != "model_params"]
            + [
                np.sum(
                    [
                        np.prod(
                            [
                                len(parameters["model_params"][m][k])
                                for k in parameters["model_params"][m].keys()
                            ]
                        )
                        for m in models
                    ]
                )
            ]
        )

        for epochs, normalization_method, past_history_factor in itertools.product(
            parameters["epochs"],
            parameters["normalization_method"],
            parameters["past_history_factor"],
        ):
            for batch_size, learning_rate in itertools.product(
                parameters["batch_size"], parameters["learning_rate"]
            ):
                
                for model_name in models:
                    
                    for model_index, model_args in enumerate(
                        product(**parameters["model_params"][model_name])
                    ):
                        
                        experiments_index += 1
                        if experiments_index <= current_index:
                            continue

                        # Run each experiment in a new Process to avoid GPU memory leaks
                        manager = Manager()
                        error_dict = manager.dict()

                        #for i in range(1):
                        
                        p = Process(
                            target=run_experiment,
                            args=(
                                error_dict,
                                gpu_device,
                                dataset,
                                dataset_path,
                                results_path,
                                csv_filepath,
                                metrics,
                                epochss,
                                normalization_method,
                                past_history_factor,
                                parameters["max_steps_per_epoch"][0],
                                batch_size,
                                learning_rate,
                                model_name,
                                model_index,
                                model_args,
                                n_train_sample
                            ),
                        )
                        p.start()
                        p.join()

                        assert error_dict["status"] == 1, error_dict["message"]

                        notify_slack(
                            "{}/{} {}:{}/{} ({})".format(
                                dataset_index + 1,
                                len(datasets),
                                dataset,
                                experiments_index,
                                num_total_experiments,
                                model_name,
                            )
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset path to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-sample",
        "--n_train_sample",
        help="Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-p", "--parameters", help="Parameters file path",
    )
    parser.add_argument(
        "-o", "--output", default="./results", help="Output path",
    )
    parser.add_argument(
        "-c", "--csv_filename", default="results.csv", help="Output csv filename",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU device")
    parser.add_argument(
        "-s",
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to use for evaluation. If not define it will use all possible metrics.",
    )
    parser.add_argument("-e", "--epochss", type=int, help="Number of epochs")
    args = parser.parse_args()
    # args = argparse.Namespace(
    #     datasets=['LSM/data/ExchangeRate'],
    #     models=['lsm-hh_psc_alpha-stdp', 'lsm-hh_psc_alpha-static', 'lsm-hh_psc_alpha-tsodyks', 'lsm-iaf_psc_exp-stdp', 'lsm-iaf_psc_exp-static', 'lsm-iaf_psc_exp-tsodyks'],
    #     parameters='LSM/experiments/parameters.json',
    #     output='./testLsm',
    #     csv_filename='testLsm.csv',
    #     gpu=0,
    #     metrics=None
    # )

    main(args)
