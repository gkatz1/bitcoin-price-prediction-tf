import os
import random
import tqdm
import argparse
import time
import datetime
from datetime import timedelta
from datetime import date
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.batch_generator import batch_generator

tf.reset_default_graph()


# TODO:
# run training on gpu

# *********** data processing ***********
def to_supervised(data, look_back=1, num_features=1):
    """
    creates inputs, labels
    returns X, Y
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0].reshape(look_back, num_features[0]))
        Y.append(data[i + look_back, 0].reshape(1, num_features[1]))  # predicting next single value
    X, Y = np.array(X), np.array(Y)
    return X, Y

def preprocess_data(data_params):

    # *************** params ******************
    look_back = data_params['look_back']
    train_set_fraction = data_params['train_set_fraction'] # 0.75
    dataset_path = data_params["dataset_path"]
    num_features = data_params['input_num_features'], data_params['output_num_features']

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
    
    X_train, y_train = to_supervised(train_set, look_back, num_features)
    X_test, y_test = to_supervised(test_set, look_back, num_features)
    
    # X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    # X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
    
    return X_train, y_train, X_test, y_test, scaler


def inverse_transforms(data, scaler):
    """
    invese transforms:
        1. scaling
        2. stationary
    """
    data = tf.reshape(data, [-1, 1])
    data_inverse = scaler.inverse_transform(data)


# *********** model ***********
def get_hyperparams():
    params = {
        'learning_rate' : 1e-3,
        'look_back' : 1,
        'prediction_step' : 1,
        'train_set_fraction' : 0.75,
        'num_epochs' : 100,
        'batch_size' : 2,
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
    num_epcohs = model_params['num_epochs']   # add early stopping, save best model
    input_num_features = model_params['input_num_features']  # 1
    output_num_features = model_params['output_num_features']
    look_back = model_params['look_back']
    n_hidden = model_params['n_hidden']
    learning_rate = model_params['learning_rate']
    val_batch_size = batch_size

    X_train, y_train, X_validation, y_validation, scaler = preprocess_data(data_params)
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
    cur_state_0_c = current_state.c
    cur_state_0_h = current_state.h
    tf.summary.histogram('rnn_outputs', outputs)

    # what is the shape of outputs, outputs[:, -1, :] ?
    prediction = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']


    with tf.name_scope('Metrices'):
        # y = tf.reshape(y, [-1]) # necesary?
        # regular version VS v2 VS sparse
        # what about the shape of prediction VS shape of y??
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        loss = tf.losses.mean_squared_error(labels=y, predictions=prediction)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # in order to calculate rmse we need to transform data back to a price
        # prediction_inverse = inverse_transforms(prediction, scaler)
        # y_inverse = inverse_transforms(y, scaler)
        # prediction_inverse = prediction_inverse.squeeze()[1:]
        # y_test_inverse = y_test_inverse.squeeze()
        # rmse = sqrt(mean_squared_error(prediction_inverse, y_inverse))
        # tf.summary.scalar('rmse', rmse)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    _cur_state_0_c = np.zeros((batch_size, n_hidden), dtype=np.float32)
    _cur_state_0_h = np.zeros((batch_size, n_hidden), dtype=np.float32)
    print(_cur_state_0_c.shape)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # training
        print("==> debugging")
        for epoch in tqdm.tqdm(range(num_epcohs)):
            i = 0
            train_rmse = 0.0
            for i, data in tqdm.tqdm(enumerate(train_loader, 0), total=num_training_batches):
                if i == num_training_batches:
                    break
                _x, _y = data
                _x = _x.reshape((-1, look_back, input_num_features))
                _y = _y.reshape((-1, output_num_features))
                _loss, _, _summary, _cur_state_0_c, _cur_state_0_h= sess.run([
                    loss, optimizer, merged_summary, cur_state_0_c, cur_state_0_h],
                    feed_dict = {
                        x: _x,
                        y: _y,
                        init_state_0_h: _cur_state_0_h,
                        init_state_0_c: _cur_state_0_c
                    }
                )
                # print(_pred[0])
                # print(np.shape(_pred))
                # print(_outputs[0].shape)
                train_writer.add_summary(_summary, i + num_training_batches * epoch)
        train_writer.close()
        saver.save(sess, model_save_path)
        print("Run 'tensorboard --logdir=./{}' to checkout tensorboard logs.".format(logs_dir_path))

def main(args):
    # ************** params **************
    dataset_path = '../../datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
    
    time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    exp_name = args.experiment_name + '_' + time_now
    
    logs_dir_path = './runs/{}/'.format(exp_name)
    writer = tf.summary.FileWriter(logs_dir_path)
    model_save_path = "models/{}/{}.ckpt".format(args.experiment_name)
 
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
        'logs_dir_path' : logs_dir_path
    }
    
    # print("==> training")
    train(train_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="different hyper parameters and configurations")
    parser.add_argument('-n', '--experiment_name', required=True,
                        help="experiment name")
    args = parser.parse_args()
    main(args)


