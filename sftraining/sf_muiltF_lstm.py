import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

#loading data
from matplotlib import pyplot as plt

from sftraining.numpy_utils import window_array_1d

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

dataset_train = sales_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0'])


dataset_test = test_data.pivot_table(index=['STOREID','PRODUCTID'],values=['TARGET'], columns=['Unnamed: 0'])
dataset = pd.merge(dataset_train,dataset_test,on = ['PRODUCTID','STOREID'],how = 'left')

#
# dataset_infer = infer_data.pivot_table(index=['STOREID','PRODUCTID'],values = ['TARGET'],columns = ['Unnamed: 0'],fill_value = 0)
# dataset = pd.merge(dataset_train,dataset_infer,on = ['PRODUCTID','STOREID'],how = 'left')

feature_others_train = sales_data.iloc[:, 1:]
feature_others_test = test_data.iloc[:, 1:]
feature_others_infer = infer_data.iloc[:, 1:]

feature_others_concat = pd.concat([feature_others_train, feature_others_test, feature_others_infer], axis=0, ignore_index=True)

sky = feature_others_concat['SKY'].values
# f_sky = sky[7:]   #用未来7天的天气加入特征
f_sky = sky[:-7]

week = feature_others_concat['Week'].values
f_week = week[:-7]

holiday = feature_others_concat['Holiday'].values
f_holiday = holiday[:-7]



# print(dataset)
series = dataset.values.flatten()
# print(dataset.values.flatten())

# def plot_series(time, series, format="-", start=0, end=None):
#     plt.plot(time[start:end], series[start:end], format)
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.grid(True)
#
#
# time = dataset['Unnamed: 0'].values
# type(time)
# print(time)

# print(type(series))
# print(type(f_sky))
window_size = 30
f_target = window_array_1d(series, window_size, 1)
f_sky = window_array_1d(f_sky, window_size, 1)
f_week = window_array_1d(f_week, window_size, 1)
f_holiday = window_array_1d(f_holiday, window_size, 1)


f_target = np.expand_dims(f_target, axis=-1)
f_sky = np.expand_dims(f_sky, axis=-1)
f_week = np.expand_dims(f_week, axis=-1)
f_holiday = np.expand_dims(f_holiday, axis=-1)


feature_all = np.concatenate((f_target, f_sky, f_week, f_holiday), axis=-1)


train_feature = feature_all[:-86, :, :]
test_feature = feature_all[-86:-1, :, :]

train_label = series[window_size:-85]
test_label = series[-85:]


# design network


model = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(train_feature.shape[1], train_feature.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(1)
])
#
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
print(model.summary())

# # train
# history = model.fit(train_feature, train_label, epochs=100, verbose=1)
# #
# model.save('../modelh5/twofeatures_lstm_save.h5')
# #
# test
model = load_model('../modelh5/twofeatures_lstm_save.h5')
test_pre = model.predict(test_feature, verbose=2)

print(test_pre)

model.evaluate(test_feature, test_label, verbose=2)
mae_Score = mean_absolute_error(test_pre, test_label)
print("The mean absolute error (MAE) on test set: {:.2f}".format(mae_Score))

test_pre = test_pre.flatten()
mape_Score = mape(test_label, test_pre)
print(mape_Score)
print('验证集/准确率: %.2f' % (100 - mape_Score) + '%')

print("end")


