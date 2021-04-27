# encoding: utf-8
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import datetime
import time
import os
from sklearn.preprocessing import MinMaxScaler

NowDate = time.strftime('%Y%m%d', time.localtime())

inbasefielDir = "./trainData/"
inbaseVfielDir = "./VerifyData/"
inbasefileName = "_trainData.csv"
inbaseVfileName = "_verifyData.csv"
outDir = "./pkl/"
outFiles = ".pkl"

StoreListFile = './ProductID_List.csv'

# StoreListCalsaveFile = 'D:/Python/Skylark/Evaluation.csv'
ErrorStoreList = []  # 有問題店號清單

if not os.path.exists(outDir):
    os.mkdir(outDir)

# 寫入Result
Result_df = pd.DataFrame(columns=["DATE", "WEEKDAY", "STOREID", "PRODUCTID",
                                  "PRODUCTNAME", "Holiday", "Celebration", "LOWTEMP", "HIGHTEMP",
                                  "SKY", "Entry", "COVID19",
                                  "PREDICTED_QTY", "ACTUAL_QTY", "ABSOLUTE_ERROR"])
# 取得店號清單CSV檔案
EV_df = pd.DataFrame(columns=["STOREID", "PRODUCTID"])
MAE_All = 0
firstFile = 1
pre_storeID = ""
df = pd.read_csv(StoreListFile, delimiter=',', dtype={'STOREID': str, 'PRODUCTID': str})
dicts = df.to_dict('records')
for index, row in df.iterrows():
    storeID = df.loc[index, 'STOREID'].rstrip()
    productID = df.loc[index, 'PRODUCTID']
    productName = df.loc[index, 'PRODUCTNAME']
    # qty=df.loc[index,'QTY']

    url = inbasefielDir + storeID + '_' + productID + inbasefileName
    train_csv = pd.read_csv(url, engine='python')

print("done")