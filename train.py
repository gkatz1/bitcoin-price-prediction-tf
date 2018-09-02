import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from random import randint
from matplotlib import pyplot
from datetime import datetime
from datetime import date
from datetime import timedelta
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras import initializers


def to_supervised(data, look_back=1):
    """
    creates inputs, labels
    returns X, Y
    """ 
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    X, Y = np.array(X), np.array(Y)
    return X, Y

def scale():
    pass


def plot_graphs(history):
    
    trace1 = go.Scatter(
        x = np.arange(0, len(history.history['loss']), 1),
        y = history.history['loss'],
        mode = 'lines',
        name = 'Train loss',
        line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(history.history['val_loss']), 1),
        y = history.history['val_loss'],
        mode = 'lines',
        name = 'Test loss',
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )

    data = [trace1, trace2]
    layout = dict(title = 'Train and Test Loss during training',
                  xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='training_process')


def prepare_data(data_params):

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


def process_data(visulize):
    data = pd.read_csv('../datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv')
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
    train_delta = 0.75 * num_samples

    from datetime import timedelta
    train_d_start = daily_price.index[0]
    train_d_end = train_d_start + timedelta(days=train_delta)
    test_d_start = train_d_end
    print(train_d_start, train_d_end, test_d_start)

    train_data = daily_price[train_d_start:train_d_end]
    test_data = daily_price[train_d_end:]
    print(len(train_data), len(test_data))

    
    if visulize:
        working_data = [train_data, test_data]
        working_data = pd.concat(working_data)

        working_data = working_data.reset_index()
        working_data['date'] = pd.to_datetime(working_data['date'])
        working_data = working_data.set_index('date')
        
        s = sm.tsa.seasonal_decompose(working_data.Weighted_Price.values, freq=60)

        trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
            line = dict(color = ('rgb(244, 146, 65)'), width = 4))
        trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
            line = dict(color = ('rgb(66, 244, 155)'), width = 2))

        trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
            line = dict(color = ('rgb(209, 244, 66)'), width = 2))

        trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
            line = dict(color = ('rgb(66, 134, 244)'), width = 2))

        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Seasonal decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='seasonal_decomposition')
        # plt.plot(fig)
        # plt.show()

def main(args):
    # ********* arguments & params********
    visualize = args.visualize
    dataset_path = '../datasets/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
   
    # process_data(visualize)
    data_params = dict() 
    data_params["dataset_path"] = dataset_path
    data_params["look_back"] = 1
    data_params["train_set_fraction"] = 0.75
    X_train, y_train, X_test, y_test, scaler = prepare_data(data_params)
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dense(1))

    # compile and fit the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, shuffle=False,
                        validation_data=(X_test, y_test),
                        callbacks = [EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=20, verbose=1)])

    plot_graphs(history)

    X_test = np.append(X_test, X_test[-1])
    X_test = np.reshape(X_test, (len(X_test), 1, 1))

    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # print("prediction_inverse")
    # print(type(prediction_inverse), prediction_inverse.shape, prediction_inverse)
    # print(" y_test_inverse")
    # print(type(y_test_inverse), y_test_inverse.shape, y_test_inverse)

    prediction_inverse = prediction_inverse.squeeze()[1:]
    y_test_inverse = y_test_inverse.squeeze()
    print(prediction_inverse.shape, y_test_inverse.shape)

    trace1 = go.Scatter(
        x = np.arange(0, len(prediction_inverse), 1),
        y = prediction_inverse,
        mode = 'lines',
        name = 'Predicted labels',
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(y_test_inverse), 1),
        y = y_test_inverse,
        mode = 'lines',
        name = 'True labels',
        line = dict(color=('rgb(66, 244, 155)'), width=2)
    )
    
    data = [trace1, trace2]
    layout = dict(title = 'Comparison of true prices (on the test dataset) with prices our model predicted',
                 xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='results_demonstrating')

    rmse = sqrt(mean_squared_error(Y_test2_inverse, prediction2_inverse))
    print("rmse = {:.3f}".format(rmse))
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="different hyper parameters and configurations")
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="visualize different plots")
    args = parser.parse_args()
    main(args)
