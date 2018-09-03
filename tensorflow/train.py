# from __future__ import print_function

import os
import random
import tqdm
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


tf.reset_default_graph()


# *********** data processing ***********
def preprocess_data(data_params):

    # *************** params ******************
    look_back = data_params['look_back']
    train_set_fraction = data_params['train_set_fraction'] # 0.75
    dataset_path = data_params["dataset_path"]

    data = pd.read_csv(dataset_path)
    # print(data.isnull().values.any())
    # print(data.head(10))

    data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
    group = data.groupby('date')
    daily_price = group['Weighted_Price'].mean()

    print(daily_price.head())
    # print(daily_price.tail())
    print(str(len(daily_price.index)))
    print(daily_price.index[0])

    num_samples = len(daily_price.index)
    train_delta = train_set_fraction * num_samples
    # train_delta = 600  # debug
    # test_delta = 60  # debug

    train_d_start = daily_price.index[0]
    train_d_end = train_d_start + timedelta(days=train_delta)
    test_d_start = train_d_end
    # test_d_end = train_d_end + timedelta(days=test_delta)  # debug
    print(train_d_start, train_d_end, test_d_start)

    train_data = daily_price[train_d_start:train_d_end]
    # test_data = daily_price[train_d_end:test_d_end]   # debu
    test_data = daily_price[train_d_end:]
    print("train_data, test_data")
    print(len(train_data), len(test_data))
    # print(train_data, test_data)

    train_set = train_data.values
    train_set = np.reshape(train_set, (len(train_set), 1))
    test_set = test_data.values
    test_set = np.reshape(test_set, (len(test_set), 1))

    scaler = MinMaxScaler() # feature_range=(-1,1)
    train_set = scaler.fit_transform(train_set) # scaler.fit() ?
    test_set = scaler.transform(test_set)
    
    X_train, y_train = to_supervised(train_set, look_back)
    X_test, y_test = to_supervised(test_set, look_back)
    
    X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
    
    return X_train, y_train, X_test, y_test, scaler


def inverse_transforms(data, scaler):
    """
    invese transforms:
        1. scaling
        2. stationary
    """
    data_inverse = scaler.inverse_transform(data.reshape(-1, 1))


# *********** model ***********
def get_hyperparams()
    params = {
        'learning_rate' : 1e-3,
        'look_back' : 1,
        'prediction_step' : 1
        'train_set_fraction' : 0.75,
        'batch_size' : 1,
        'n_hidden' : 256
    }
    return params


# TODO
# what are the shapes of: x, init_state, outputs, outputs[:, -1, :]
def model(x, init_state):
    # x is of shape e.g. [16, 5]
    # y, pred are of shape e.g. [16, 1]
    n_hidden = model_params['n_hidden']   # 256 ?
    learning_rate = model_params['learning_rate']
    scaler = model_params['scaler']
    num_layers = model_params['num_layers']

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, prediction_num_features]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([prediction_num_features]))
    }
    
    rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    if reset_state:
        # what is the shape of init_state?
        init_state = cell.zero_state(batch_size, tf.float32)
    # make sure 'x' is of the right shape
    outputs, current_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)
    tf.summary.histogram('rnn_outputs', outputs)

    # what is the shape of outputs, outputs[:, -1, :] ?
    prediction = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']

    return prediction, current_state


# ************* training *************
def train(train_params):
    # ************** params **************
    
    debug = True
    
    data_params = train_params['data_params']
    batch_size = train_params['batch_size']
    num_epcohs = train_params['num_epochs']   # add early stopping, save best model
    input_num_features = train_params['input_num_features']  # 1
    output_num_features = train_params['output_num_features']
    
    X_train, y_train, X_validation, y_validation, scaler = preprocess_data(data_params)
    train_loader = batch_generator(X_train, y_train, batch_size)
    val_loader = batch_generator(X_validation, y_validation, val_batch_size)

    train_writer = tf.summary.FileWriter(os.path.join(logs_dir_path, 'train')) #, accuracy.graph)
    val_writer = tf.summary.FileWriter(os.path.join(logs_dir_path, 'validation')) #, accuracy.graph)

    saver = tf.train.Saver()
    
    num_training_batches = X_train.shape[0] // batch_size
    num_validation_batches = X_validation.shape[0] // batch_size

    # let's say prices are: [1,2,3,4,5,6,7,8,9,10,11,12]
    # and we use the prices from the past 3 minutes/days (look back = 3)
    # in order to predict the next
    # let batch_size = 3 and let num_batches = 3
    # input: [[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
    #        [[4,5,6], [5,6,7], [6,7,8]]
    #        [7,8,9], [8,9,10], [9,10,11]]]
    # labels: [[[4], [5], [6]]
    #         [[7], [8], [9]]
    #         [[10], [11], [12]]]
    #
    # Visualizations help!
    
    x = tf.placeholder(tf.float32, [None, look_back, input_num_features]) # 1 feature (price)
    y = tf.placeholder(tf.float32, [None, output_num_features])
    # WHAT IS THE SHAPE OF init_state ?
    init_state = None # tf.placeholder(??)
    
    reset_state = tf.placeholder(tf.bool)
    
    prediction, current_state = model(x, init_state)
    
    with tf.name_scope('Metrices')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        # in order to calculate rmse we need to transform data back to a price
        prediction_inverse = inverse_transforms(prediction, scaler)
        y_inverse = inverse_transforms(y, scaler)
        prediction_inverse = prediction_inverse.squeeze()[1:]
        y_test_inverse = y_test_inverse.squeeze()
        rmse = sqrt(mean_squared_error(prediction_inverse, y_inverse))
        tf.summary.scalar('rmse', rmse)
    
    merged_summary = tf.summary.merge_all()
    
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        # training
        print("==> training")
        for epoch in tqdm.tqdm(range(num_epcohs)):
            train_rmse = 0.0
            for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=num_training_batches):
                _x,_y = data
                _loss, _rmse, _, _summary, _current_state = sess.run([loss, rmse, optimizer,
                    merged_summary, current_state],
                    feed_dict = {
                    x: _x,
                    y = _y,
                    current_state: _current_state,
                    reset_state: True if i == 0 else False
                    })
                    
                train_rmse += _rmse
                train_writer.add_summary(_summary, i + num_training_batches * epoch)

        # validation
        print("==> validation")
        for epoch in tqdm.tqdm(range(num_epcohs)):
            for i, data in tqdm.tqdm(enumerate(val_loader, 0), total=num_training_batches):
                _x, _y = data
                _rmse, _summary, _current_state = sess.run([rmse, merged_summary, current_state],
                    feed_dict = {
                    x: _x,
                    y: _y
                    current_state: _current_state
                    })
                val_writer.add_summary(_summary, i + num_validation_batches * epoch)


def main(args):
    # ************** params **************
    dataset_path = '../datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
    
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    exp_name = args.experiment_name + '_' + time_now
    
    logs_dir_path = '/tmp/tensorflow/{}/'.format(exp_name)
    writer = tf.summary.FileWriter(logs_path)
    
    hyper_params = get_hyperparams()
    data_params = dict()
    data_params["dataset_path"] = dataset_path
    data_params["look_back"] = hyper_params["look_back"]
    data_params["train_set_fraction"] = hyper_params["train_set_fraction"]

    train_params = {
        'data_params' : data_params,
        'batch_size' : hyper_params['batch_size']
    }
    
    # print("==> training")
    train(train_params)




# is input of e.g. size = 3 means that we input each sample into one cell
# or that we input a vector of size 3 into one cell?
# 1 cell, vector of 3 as input, vector of size 3 as an output, and we roll it
# num_steps times, where input now will be the output of the prev?
# no. We have Z vectors, each of size X inputted into an unrolled version of Z cells
                                                     
                                            


