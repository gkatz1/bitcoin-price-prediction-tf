import os
import random
import tqdm
import argparse
import time
import datetime
from datetime import timedelta
from datetime import date
import datetime as dt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt

from utils.batch_generator import batch_generator

tf.reset_default_graph()


# TODO:
# run training on gpu

# *********** data processing ***********
def to_supervised(data, look_back=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, look_back + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    print("inverse_difference")
    print("history.shape ={}".format(history.shape))
    print("yhat.shape ={}".format(yhat.shape))
    return yhat + history[-interval]

# scale
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# TODO: change names, etc
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def timestamp_parser(x):
    return pd.to_datetime(x, unit='s')


def preprocess_data(data_params):

    # *************** params ******************
    look_back = data_params['look_back']
    train_set_fraction = data_params['train_set_fraction'] # 0.75
    test_set_fraction = 1 - train_set_fraction
    dataset_path = data_params["dataset_path"]
    num_features = data_params['input_num_features'], data_params['output_num_features']


    b_data = pd.read_csv(dataset_path, usecols=['Timestamp', 'Close'], 
        squeeze=True, index_col=0, parse_dates=[0], date_parser=timestamp_parser)


    # start = b_data.index.searchsorted(dt.datetime(2018, 1, 2))
    # end = b_data.index.searchsorted(dt.datetime(2018, 1, 6))
    start = b_data.index.searchsorted(dt.datetime(2018, 1, 2))
    end = b_data.index.searchsorted(dt.datetime(2018, 1, 6))
    b_data = b_data[start:end]
    num_samples = len(b_data.index)
    raw_values = b_data.values

    print(raw_values)
    print("min = {}, max = {}, num samples = {}".format(
        min(raw_values), max(raw_values), num_samples))
    # raise NotImplementedError()
 
    print(b_data.head())
    print(num_samples)
    print(raw_values)
    print(type(raw_values))
    
    # train_start_idx = 0
    # train_end_idx = int(train_set_fraction * num_samples)
  
    # new logic

    diff_vals = difference(raw_values, look_back)
    supervised = to_supervised(diff_vals, look_back)

    supervised_vals = supervised.values
    train = supervised_vals[0:-int(num_samples * test_set_fraction)] 
    test = supervised_vals[-int(num_samples * test_set_fraction):]
    scaler, train_scaled, test_scaled = scale(train, test)

    data_params['training_set_size'] = max(train.shape)
    data_params['validation_set_size'] = max(test.shape)
    print("training set size = {}, validation set size = {}".format(
        max(train.shape), max(test.shape)))

    train_x, train_y = train_scaled[:, 0:-1], train_scaled[:, -1]
    test_x = test_scaled[:, 0:-1]
    test_y = raw_values[-int(num_samples * test_set_fraction):]
    with open("train_x", 'w') as f:
        for el in train_x:
            f.write("%3f " % el)
    with open("train_y", 'w') as f:
        for el in train_y:
            f.write("%3f " % el)
    with open("test_x", 'w') as f:
        for el in test_x:
            f.write("%3f " % el)
    with open("test_y", 'w') as f:
        for el in test_y:
            f.write("%3f " % el)
    # raise NotImplementedError()

    print("test_x.shape = {}, test_y.shape = {}".format(
        test_x.shape, test_y.shape))
    
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    print(train_x)
    # raise NotImplementedError()

    # TBC to a more elegant way
    train_x = train_x.reshape([max(train_x.shape), look_back, num_features[0]])
    test_x = test_x.reshape([max(test_x.shape), look_back, num_features[0]])
    train_y = train_y.reshape([max(train_y.shape), look_back, num_features[1]])
    # test_y = test_y.reshape([max(test_y.shape), look_back, num_features[1]])
 
    return raw_values, train_x, train_y, test_x, test_y, scaler


def inverse_transforms(data, scaler):
    """
    invese transforms:
        1. scaling
        2. stationary
    """
    # print("inverse_transforms(), data")
    # print(data)

    data = data.reshape(-1, 1)
    ## data = tf.reshape(data, [-1, 1])
    # data = tf.reshape(data, [-1])
    # data = tf.squeeze(data, axis=0)
    data_inverse = scaler.inverse_transform(data)
    return data_inverse

# *********** model ***********
def get_hyperparams():
    params = {
        'learning_rate' : 1e-3,
        'look_back' : 1,
        'prediction_step' : 1,
        'train_set_fraction' : 0.75,
        'num_epochs' : 30,
        'batch_size' : 1,
        'n_hidden' : 256,
        'num_layers' : 3,
        'input_num_features' : 1,
        'output_num_features' : 1
    }
    return params


def train(train_params):

    debug = True
    
    data_params = train_params['data_params']
    model_params = train_params['model_params']
    model_save_path = train_params['model_save_path']
    logs_dir_path = train_params['logs_dir_path']
    batch_size = model_params['batch_size']
    num_epochs = model_params['num_epochs']   # add early stopping, save best model
    input_num_features = model_params['input_num_features']  # 1
    output_num_features = model_params['output_num_features']
    look_back = model_params['look_back']
    n_hidden = model_params['n_hidden']
    learning_rate = model_params['learning_rate']
    val_batch_size = batch_size
    visualize = train_params['visualize']

    raw_values, X_train, y_train, X_validation, y_validation, scaler = preprocess_data(data_params)
    print(raw_values.shape, X_train.shape, y_train.shape, X_validation.shape, y_validation.shape)
 
    # print(X_train[0])
    # print(X_train[0].shape)
    # print(X_train[0:2].shape)
    # gen = batch_generator(X_train, y_train, batch_size)
    # print(next(gen)[0].shape)
    # print(next(gen)[1].shape)

    train_loader = batch_generator(X_train, y_train, batch_size)
    val_loader = batch_generator(X_validation, y_validation, val_batch_size)

    train_writer = tf.summary.FileWriter(os.path.join(logs_dir_path, 'train')) #, accuracy.graph)
    val_writer = tf.summary.FileWriter(os.path.join(logs_dir_path, 'validation')) #, accuracy.graph)

    num_training_batches = X_train.shape[0] // batch_size
    num_validation_batches = X_validation.shape[0] // batch_size
    print("num_training_batches: {}, num_validation_batches: {}".format(
        num_training_batches, num_validation_batches))

    x = tf.placeholder(tf.float32, [None, look_back, input_num_features]) # 1 feature (price)
    y = tf.placeholder(tf.float32, [None, output_num_features])

    init_state_0_c = tf.placeholder(tf.float32, [None, n_hidden])
    init_state_0_h = tf.placeholder(tf.float32, [None, n_hidden])

    # **** model ****
    n_hidden = model_params['n_hidden']
    learning_rate = model_params['learning_rate']
    num_layers = model_params['num_layers']
    prediction_num_features = model_params['output_num_features']
    batch_size = model_params['batch_size']

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, prediction_num_features]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([prediction_num_features]))
    }

    rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)

    rnn_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(init_state_0_c, init_state_0_h)

    outputs, current_state = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=rnn_tuple_state)
    # cur_state_0_c = current_state.c
    # cur_state_0_h = current_state.h

    tf.summary.histogram('rnn_outputs', outputs)

    prediction = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']


    with tf.name_scope('Metrices'):
        # y = tf.reshape(y, [-1]) # necesary?
        # regular version VS v2 VS sparse
        # what about the shape of prediction VS shape of y??
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)  # is the loss calculated correctly? are these the params I need to pass?
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # in order to calculate rmse we need to transform data back to a price
        ## prediction_inverse = inverse_transforms(prediction, scaler)
        ## y_inverse = inverse_transforms(y, scaler)
        ## prediction_inverse = prediction_inverse.squeeze()[1:]
        ## y_test_inverse = y_test_inverse.squeeze()
        ## rmse = sqrt(mean_squared_error(prediction_inverse, y_inverse))
        ## tf.summary.scalar('rmse', rmse)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    _cur_state_0_c = np.zeros((batch_size, n_hidden), dtype=np.float32)
    _cur_state_0_h = np.zeros((batch_size, n_hidden), dtype=np.float32)
    _current_state = tuple((_cur_state_0_c, _cur_state_0_h))

    ## print(_cur_state_0_c.shape)
    print("Run 'tensorboard --logdir=./{}' to checkout tensorboard logs.".format(logs_dir_path))
    print("==> training")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # training
        for epoch in tqdm.tqdm(range(num_epochs)):
            train_rmse = 0.0
            for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=num_training_batches):
                # print(i)
                if i == num_training_batches:
                    print("*" * 50)
                    break
                _x, _y = data
                _x = _x.reshape((-1, look_back, input_num_features))
                _y = _y.reshape((-1, output_num_features))
                _pred, _loss, _, _summary, _current_state = sess.run([
                    prediction, loss, optimizer, merged_summary, current_state],
                    feed_dict = {
                        x: _x,
                        y: _y,
                        init_state_0_c: _current_state[0],
                        init_state_0_h: _current_state[1]
                    }
                )


                train_writer.add_summary(_summary, i + num_training_batches * epoch)

        train_writer.close()
        saver.save(sess, model_save_path)

        validation_set_size = data_params['validation_set_size']
        train_set_size = data_params['training_set_size']
        predictions = list()
        ground_truths = list()

        print("==> validating")
        # print(num_validation_batches)
        for epoch in tqdm.tqdm(range(1)):
            total_rmse = 0.0
            for i, data in tqdm.tqdm(enumerate(val_loader, 0), total=num_validation_batches):
                if i == 0:
                    history = []
                if i == num_validation_batches:
                    break
                _x, _y = data
                _x = _x.reshape((-1, look_back, input_num_features))
                _y = _y.reshape((-1, output_num_features))

                _pred, _summary, _current_state = sess.run([
                    prediction, merged_summary, current_state],
                    feed_dict = {
                        x: _x,
                        y: _y,
                        init_state_0_c: _current_state[0],
                        init_state_0_h: _current_state[1]
                    }
                )

                # print(_pred.shape)
                # print(type(_pred))
                # print(_pred)
                # _pred = np.squeeze(_pred, axis=0)
                # print(_pred, _pred.shape)
                # prediction_inverse = inverse_transforms(_pred, scaler)
                prediction_inverse = invert_scale(scaler, _x, _pred)
                # prediction_inverse = prediction_inverse.squeeze(axis=0)
                prediction_inverse = inverse_difference(raw_values, prediction_inverse, \
                    validation_set_size + 1 - i)
                predictions.append(prediction_inverse)
                ground_truths.append(raw_values[train_set_size + i + 1])    # why +1 ?
                # rmse = sqrt(mean_squared_error(prediction_inverse, _y))
                print("predicted = {}, groud truth = {}".format(prediction_inverse, \
                    raw_values[train_set_size + i + 1]))

                # print("rmse: {}".format(rmse))
                # print("\n")
                # total_rmse += rmse

        val_writer.close()

        # total_rmse /= num_validation_batches
        # rmse_2 = sqrt(mean_squared_error(ground_truths, predictions))
        predictions = np.array(predictions)
        print("predictions.shape = {}, y_validation.shape = {}".format(
            predictions.shape, y_validation.shape))
        rmse_2 = sqrt(mean_squared_error(y_validation, predictions))

        # print("validation rmse: {}".format(total_rmse))
        print("validation rmse 2: {}".format(rmse_2))
        if visualize:
            plt.plot(ground_truths, 'r')
            plt.plot(predictions, 'b')
            plt.show()


def main(args):
    # ************** params **************
    dataset_path = '../../datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
    
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    exp_name = args.experiment_name + '_' + time_now
    visualize = args.visualize
 
    logs_dir_path = './runs/{}/'.format(exp_name)
    writer = tf.summary.FileWriter(logs_dir_path)
    model_save_path = "models/{0}/{0}.ckpt".format(args.experiment_name)
 
    hyper_params = get_hyperparams()
    data_params = dict()
    data_params["dataset_path"] = dataset_path
    data_params["look_back"] = hyper_params["look_back"]
    data_params["train_set_fraction"] = hyper_params["train_set_fraction"]
    data_params['input_num_features'] = hyper_params['input_num_features']
    data_params['output_num_features'] = hyper_params['output_num_features']

    train_params = {
        'data_params' : data_params,
        'model_params' : hyper_params,
        'model_save_path' : model_save_path,
        'logs_dir_path' : logs_dir_path,
        'visualize' : visualize
    }
    
    # print("==> training")
    train(train_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="different hyper parameters and configurations")
    parser.add_argument('-n', '--experiment_name', required=True,
                        help="experiment name")
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="visualize results")
    args = parser.parse_args()
    main(args)


