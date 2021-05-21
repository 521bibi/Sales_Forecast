import math
import os
import time


import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

from utils.crud import find2pd, connect_mongo

inbaseTfileName = "_trainData"
inbaseVfileName = "_validationData"
outDir = "./pkl/"
outFiles = ".pkl"

StoreListFile = './ProductID_List.csv'

if not os.path.exists(outDir):
    os.mkdir(outDir)


def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

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



    collection_train_name = storeID + '_' + productID + inbaseTfileName
    collection_train = connect_mongo(collection_train_name)
    train_DF = find2pd(collection_train)

    collection_validation_name = storeID + '_' + productID + inbaseVfileName
    collection_validation = connect_mongo(collection_validation_name)
    validation_DF = find2pd(collection_validation)

    if len(train_DF) > 1:
        outFilename = outDir + storeID + '_' + productID + outFiles

        train_DF = train_DF.iloc[:, 1:]
        X_train = train_DF.iloc[:, :-1]
        y_train = train_DF.iloc[:, -1]

        # scaler = MinMaxScaler()
        # scaler.fit(train_DF)

        params = {'n_estimators': 100,
                  'max_depth': 10,
                  # 'min_samples_split': 5,
                  # 'learning_rate': 0.05,
                  'loss': 'lad'}
        # 梯度boosting
        reg = GradientBoostingRegressor(**params)
        reg.fit(X_train, y_train)

        validation_DF = validation_DF.iloc[:, 1:]
        X_validation = validation_DF.iloc[:, :-1]
        y_validation = validation_DF.iloc[:, -1].apply(pd.to_numeric, errors='ignore')

        yhat_validation = reg.predict(X_validation)

        # 评价指标
        mse_Score = mean_squared_error(y_validation, yhat_validation)
        print("The mean squared error (MSE) on test set: {:.2f}".format(mse_Score))
        rmse_Score = math.sqrt(mse_Score)
        print('验证集RMSE: %.2f' % (rmse_Score))

        mae_Score = mean_absolute_error(y_validation, yhat_validation)
        print("The mean absolute error (MAE) on test set: {:.2f}".format(mae_Score))

        mape_Score = mape(y_validation, yhat_validation)
        print('验证集/准确率: %.2f' % (100-mape_Score) + '%')

        R2 = r2_score(y_validation, yhat_validation)
        print('R2: %.2f' % (R2))

        n = train_DF.shape[0]
        p = train_DF.shape[1] - 1
        Adjusted_R2 = 1 - ((1 - r2_score(y_validation, yhat_validation)) * (n - 1)) / (n - p - 1)
        print('Adjusted_R2: %.2f' % (Adjusted_R2))


        # loss 绘制
        valid_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(reg.staged_predict(X_validation)):
            valid_score[i] = reg.loss_(y_validation, y_pred)

        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.title('Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
                 label='train Set Deviance')
        plt.plot(np.arange(params['n_estimators']) + 1, valid_score, 'r-',
                 label='validation Set Deviance')
        plt.legend(loc='upper right')
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        fig.tight_layout()
        plt.show()

        # 特征影响 查看
        feature_names = list(X_validation)
        feature_importance = reg.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.title('Feature Importance (MDI)')

        result = permutation_importance(reg, X_validation, y_validation, n_repeats=10,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        plt.subplot(1, 2, 2)
        plt.boxplot(result.importances[sorted_idx].T,
                    vert=False, labels=np.array(feature_names)[sorted_idx])
        plt.title("Permutation Importance (validation set)")
        fig.tight_layout()
        plt.show()


        # 保存模型
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "產生品號:" + storeID + "_" + productID + "預測模型檔案 ")
        # #保存Model(注:save文件夹要预先建立，否则会报错)
        joblib.dump(reg, outFilename)
