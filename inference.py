# encoding: utf-8
import json
import math

import joblib
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

from utils.crud import find2pd
from utils.featuretrans import FeatureTbyproduct


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


class Test(object):
    def __init__(self, store_id, product_id):
        # self.data_file = data_file
        self.storeID = store_id
        self.productID = product_id
        self.inbaseTfileName = "_test"
        self.outDir = "./pkl/"
        self.outFiles = ".pkl"
        self.x_fields = ["xxx", "xxx", "xxx"]
        self.x_features, self.y_labels = self.load_data()

    def load_data(self):
        # ......
        collection_test = self.storeID + '_' + self.productID + self.inbaseTfileName
        test_df = find2pd(collection_test)
        test_df = test_df.iloc[:, 1:]
        x_features = test_df.iloc[:, :-1].apply(pd.to_numeric, errors='ignore')
        y_labels = test_df.iloc[:, -1].apply(pd.to_numeric, errors='ignore')
        return x_features, y_labels

    def load_feature_names(self):
        pass

    def test_reg(self):
        modelFilename = self.outDir + self.storeID + '_' + self.productID + self.outFiles
        model = joblib.load(modelFilename)
        y_pred = model.predict(self.x_features)

        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler("log.txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        logger.addHandler(handler)
        logger.addHandler(console)

        logger.info("Start print log")
        logger.debug("Do something")
        logger.warning("Something maybe fail.")
        logger.info("Finish")
        mae = mean_squared_error(self.y_labels, y_pred)
        # print("mean_squared_error: %.2f" % mae)
        # print("RMSE: %.2f" % math.sqrt(mae))
        mape_score = mape(self.y_labels, y_pred)
        # print('验证集准确率: %.2f' % (100-mape_score) + '%')
        logger.info("mean_squared_error: %.2f" % mae)
        logger.info("mean_squared_log_error: %.2f" % mean_squared_log_error(self.y_labels, y_pred))
        logger.info("RMSE: %.2f" % math.sqrt(mae))
        logger.info("准确度: %.2f" % (100-mape_score) + '%')

        # plot_training_deviance(clf=model, n_estimators=model.get_params()["n_estimators"], X_test=self.x_features,
        #                        y_test=self.y_labels)
        #
        # # 输出feature重要性
        # logging.info("feature_importances_: %s" % model.feature_importances_)
        # plot_feature_importance(clf=model, feature_names=self.x_fields)

        return y_pred


class InferencebyProduct(object):
    def __init__(self, store_id, product_id):
        self.storeID = store_id
        self.productID = product_id
        self.inbasefileName = '_infData'
        self.outDir = "./pkl/"
        self.outFiles = ".pkl"
        self.df = self.load_data()
        self.y_pre = self.predict()

    def load_data(self):
        # ......
        collection_name = self.storeID + '_' + self.productID + self.inbasefileName
        link_collection = collection_name
        df = find2pd(link_collection)
        return df

    def predict(self):
        test = FeatureTbyproduct(self.df)
        x = test.featureT()

        modelFilename = self.outDir + self.storeID + '_' + self.productID + self.outFiles
        model = joblib.load(modelFilename)
        y_pred = model.predict(x)

        return y_pred

    def forecasting_out(self):
        forecasting_df = self.df.iloc[:, 1:]
        y = self.y_pre

        forecasting_df['predict'] = y
        forecasting_df = forecasting_df.iloc[:, 1:]
        forecasting_df = forecasting_df.rename(index={0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth', 5: 'sixth', 6: 'seventh'})

        return forecasting_df



if __name__ == '__main__':
    # store_id, product_id = '12', '141691'
    # inbasefileName = '_infData'
    # test = Test(store_id, product_id)
    # test.test_reg()

    store_id, product_id = '12', '141691'
    test = InferencebyProduct(store_id, product_id)
    x = test.forecasting_out()
    out = x.to_json(orient="index", force_ascii=False)
    parsed = json.loads(out)
    print(x)
    print(out)
    print(parsed)
