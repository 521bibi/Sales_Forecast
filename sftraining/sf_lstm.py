import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

#loading data
from matplotlib import pyplot as plt

def mape(y_true, y_pred):
    """
    返回:
    mape -- MAPE 评价指标
    """
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

os.listdir('../tmp')
sales_data = pd.read_csv('../tmp/12_141691_trainData.csv')
test_data = pd.read_csv('../tmp/12_141691_validationData.csv')

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

dataset = sales_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0','SKY'],fill_value = 0)
print(dataset)
# trainingdata = dataset.reset_index(inplace = True)
# print(dataset.head())
test_data = test_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0','SKY'],fill_value = 0)
dataset = pd.merge(dataset,test_data,on = ['PRODUCTID','STOREID'],how = 'left')
print(dataset)
series = dataset.values.flatten()
print(dataset.values.flatten())

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


# time = dataset['Unnamed: 0'].values
# type(time)
# print(time)

window_size = 30
batch_size = 16
shuffle_buffer_size = 200

# X
series_train = series[:638]
#
series_test = series[638-window_size:]

# shuffle
# def windowed_dataset(window_series, window_size, batch_size, shuffle_buffer):
#     dataset = tf.data.Dataset.from_tensor_slices(window_series)
#     dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
#     dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
#     dataset = dataset.batch(batch_size).prefetch(1)
#     return dataset

# 无shuffle
def windowed_dataset(window_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(window_series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_dataset = windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)
print(train_dataset)

test_dataset = windowed_dataset(series_test, window_size, batch_size, shuffle_buffer_size)
print(test_dataset)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
print(model.summary())

# train
# history = model.fit(train_dataset, epochs=100, verbose=1)
#
# model.save('../modelh5/simple_lstm_save.h5')

# test
model = load_model('../modelh5/simple_lstm_save.h5')
test_pre = model.predict(test_dataset, verbose=2)
model.evaluate(test_dataset, verbose=2)

# print(test_dataset[1])
test_y = np.array([])
for x,y in test_dataset:
    test_y = np.hstack([test_y, y])
print(test_y)

mae_Score = mean_absolute_error(test_pre, test_y)
print("The mean absolute error (MAE) on test set: {:.2f}".format(mae_Score))
test_pre = test_pre.flatten()
mape_Score = mape(test_y, test_pre)
print(mape_Score)
print('验证集/准确率: %.2f' % (100 - mape_Score) + '%')

print("end")

