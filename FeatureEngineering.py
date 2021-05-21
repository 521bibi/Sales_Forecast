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

from utils.crud import connect_mongo, findFromMongoDB, find2pd, DF2MongoDB, connect_mongo_db

NowDate = time.strftime('%Y%m%d', time.localtime())

inbasefileName = "_datain"
inbaseTfileName = "_trainData"
inbaseVfileName = "_validationData"
tmpDir = "./tmp/"
outDir = "./pkl/"
outFiles = ".pkl"

StoreListFile = './ProductID_List.csv'

# StoreListCalsaveFile = 'D:/Python/Skylark/Evaluation.csv'
ErrorStoreList = []  # 有問題店號清單

if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)

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

    collection_name = storeID + '_' + productID + inbasefileName
    link_collection = connect_mongo_db('FIsalesForecast', collection_name)
    train_csv = find2pd(link_collection)

    if len(train_csv) > 1:
        outFilename = outDir + storeID + '_' + productID + outFiles

        inputDate = train_csv["SDATE"]
        Year = []
        Month = []
        Day = []
        Week = []

        for i in inputDate:
            dt = datetime.datetime.strptime(i, "%Y/%m/%d")
            Year.append(dt.strftime("%Y"))
            Month.append(dt.strftime("%m"))
            Day.append(dt.strftime("%d"))
            Week.append(dt.isoweekday())

        label_encoder = preprocessing.LabelEncoder()
        Holiday = label_encoder.fit_transform(train_csv["HOLIDAY"])
        Celebration = label_encoder.fit_transform(train_csv["CELEBRATION"])

        train_storeID = train_csv["STOREID"]
        train_productID = train_csv["PRODUCTID"]

        # Weather
        encoded_HIGHTEMP = label_encoder.fit_transform(train_csv["HIGH_TEMP"])
        encoded_LOWTEMP = label_encoder.fit_transform(train_csv["LOW_TEMP"])
        encoded_SKY = label_encoder.fit_transform(train_csv["SKY"])

        # COVID-19
        Entry = train_csv["ENTRY"]
        COVID19 = train_csv["COVID19"]

        # OnSale
        encoded_ONSALE = label_encoder.fit_transform(train_csv["ONSALE"])

        datain_Target = train_csv["TARGET"]

        datain_DF = pd.DataFrame([
            pd.Series(Year, name="Year")
            , pd.Series(Month, name="Month")
            , pd.Series(Day, name="Day")
            , pd.Series(Week, name="Week")
            , pd.Series(Holiday, name="Holiday")
            , pd.Series(Celebration, name="Celebration")
            , pd.Series(encoded_LOWTEMP, name="LOWTEMP")
            , pd.Series(encoded_HIGHTEMP, name="HIGHTEMP")
            , pd.Series(encoded_SKY, name="SKY")
            , pd.Series(train_storeID)
            , pd.Series(train_productID)
            , pd.Series(Entry)
            , pd.Series(COVID19)
            # ,pd.Series(encoded_ONSALE)
            , pd.Series(datain_Target)
        ]).T

        train_DF = datain_DF.iloc[:638]
        train_Feature = train_DF.iloc[:, :-1]
        train_Target = train_DF.iloc[:, -1]
        train_DF_collection = storeID + '_' + productID + inbaseTfileName  # 集合名
        train_DF_csv = tmpDir + train_DF_collection + '.csv'   # tmp CSV
        train_DF.to_csv(train_DF_csv)  # 保存CSV
        set_teain = connect_mongo(train_DF_collection)
        DF2MongoDB(train_DF, set_teain)  # 入库保存


        validation_DF = datain_DF.iloc[638:-7]
        validation_Feature = validation_DF.iloc[:, :-1]
        validation_Target = validation_DF.iloc[:, -1]
        validation_DF_collection = storeID + '_' + productID + inbaseVfileName   # 集合名
        validation_DF_csv = tmpDir + validation_DF_collection + '.csv'  # tmp CSV
        validation_DF.to_csv(validation_DF_csv)  # 保存CSV
        set_validation = connect_mongo(validation_DF_collection)
        DF2MongoDB(validation_DF, set_validation)  # 入库保存

        test_DF = datain_DF.iloc[-7:]
        test_Feature = test_DF.iloc[:, :-1]
        test_Target = test_DF.iloc[:, -1]
        test_Feature_collection = storeID + '_' + productID + '_test'  # 集合名
        test_Feature_csv = tmpDir + test_Feature_collection + '.csv'   # tmp CSV
        test_DF.to_csv(test_Feature_csv)  # 保存CSV
        set_test = connect_mongo(test_Feature_collection)
        DF2MongoDB(test_DF, set_test)  # 入库保存


print("done")