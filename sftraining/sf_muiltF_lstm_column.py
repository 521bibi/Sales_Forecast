import numpy as np
import pandas as pd
import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model

#loading data
from matplotlib import pyplot as plt

os.listdir('../tmp')
sales_data = pd.read_csv('../tmp/12_141691_trainData.csv')
test_data = pd.read_csv('../tmp/12_141691_validationData.csv')
infer_data = pd.read_csv('../tmp/12_141691_test.csv')

def basic_eda(df):
    print("----------TOP 5 RECORDS--------")
    print(df.head(5))
    print("----------INFO-----------------")
    print(df.info())
    print("----------Describe-------------")
    print(df.describe())
    print("----------Columns--------------")
    print(df.columns)
    print("----------Data Types-----------")
    print(df.dtypes)
    print("-------Missing Values----------")
    print(df.isnull().sum())
    print("-------NULL values-------------")
    print(df.isna().sum())
    print("-----Shape Of Data-------------")
    print(df.shape)

# print("=============================Sales Data=============================")
# basic_eda(sales_data)
# print("=============================Test data=============================")
# basic_eda(test_data)

dataset = sales_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0', 'SKY'],fill_value = 0)

def setcolumn(columns, column_index, column_sky):

    # def getcolumns(columns):
    #     for item in columns:
    #         rol = len(item) - 1
    #         print(column_index)
    #         for i in range(2):
    #             column_index[0].append(item[i+1])
    #
    # getcolumns(columns=columns)
    # 列索引
    # column_index = []
    for item in columns:
        column_index.append(item[1])

    # 天气
    # column_sky = []
    for item in columns:
        column_sky.append(item[2])

    return column_index, column_sky


# trainingdata = dataset.reset_index(inplace = True)
# print(dataset.head())
test_data = test_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0','SKY'],fill_value = 0)
dataset = pd.merge(dataset,test_data,on = ['PRODUCTID','STOREID'],how = 'left')

train_columns = dataset.columns.tolist()
column_index = []
column_sky = []
getcolumns = setcolumn(train_columns, column_index, column_sky)
print(getcolumns[1])


infer_data = infer_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'], columns =['Unnamed: 0'], dropna=False)
dataset = pd.merge(dataset,infer_data,on = ['PRODUCTID','STOREID'],how = 'left')
#
print(infer_data)
print(dataset)
# series = dataset.values.flatten()
# print(dataset.values.flatten())
#
# def plot_series(time, series, format="-", start=0, end=None):
#     plt.plot(time[start:end], series[start:end], format)
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.grid(True)
#
#
# # time = dataset['Unnamed: 0'].values
# # type(time)
# # print(time)
#
# window_size = 30
# batch_size = 16
# shuffle_buffer_size = 200
#
# # X
# series_train = series[:638]
# #
# series_test = series[638-window_size:]
#
# def windowed_dataset(window_series, window_size, batch_size, shuffle_buffer):
#     dataset = tf.data.Dataset.from_tensor_slices(window_series)
#     dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
#     dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
#     dataset = dataset.batch(batch_size).prefetch(1)
#     return dataset
#
#
# train_dataset = windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)
# print(train_dataset)
#
# test_dataset = windowed_dataset(series_test, window_size, batch_size, shuffle_buffer_size)
# print(test_dataset)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
#                       input_shape=[None]),
#     # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(1)
# ])
#
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# print(model.summary())
#
# # train
# # history = model.fit(train_dataset, epochs=100, verbose=1)
# #
# # model.save('../modelh5/simple_lstm_save.h5')
#
# # test
# model = load_model('../modelh5/simple_lstm_save.h5')
# test_y = model.predict(test_dataset, verbose=2)
#
# print("test_y[0]")
#
# model.evaluate(test_dataset, verbose=2)
