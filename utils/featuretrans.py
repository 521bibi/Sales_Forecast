import datetime
import time
import pandas as pd
from sklearn import preprocessing

from utils.crud import find2pd


class FeatureTbyproduct(object):
    def __init__(self, df):
        self.df = df
        self.x_features = self.featureT()

    def featureT(self):
        feature_df = pd.DataFrame()
        if len(self.df) > 1:
            inputDate = self.df["SDATE"]
            Year = []
            Month = []
            Day = []
            Week = []

            for i in inputDate:
                dt = datetime.datetime.strptime(i, "%Y-%m-%d")           #  日期表示方法不一致了，注意
                Year.append(dt.strftime("%Y"))
                Month.append(dt.strftime("%m"))
                Day.append(dt.strftime("%d"))
                Week.append(dt.isoweekday())

            label_encoder = preprocessing.LabelEncoder()
            Holiday = label_encoder.fit_transform(self.df["HOLIDAY"])
            Celebration = label_encoder.fit_transform(self.df["CELEBRATION"])

            train_storeID = self.df["STOREID"]
            train_productID = self.df["PRODUCTID"]

            # Weather
            encoded_HIGHTEMP = label_encoder.fit_transform(self.df["HIGH_TEMP"])
            encoded_LOWTEMP = label_encoder.fit_transform(self.df["LOW_TEMP"])
            encoded_SKY = label_encoder.fit_transform(self.df["SKY"])

            # COVID-19
            Entry = self.df["ENTRY"]
            COVID19 = self.df["COVID19"]

            # OnSale
            encoded_ONSALE = label_encoder.fit_transform(self.df["ONSALE"])

            # datain_Target = self.df["TARGET"]

            feature_df = pd.DataFrame([
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
                # , pd.Series(datain_Target)
            ]).T

        return feature_df

    def load_feature_names(self):
        pass

    def labelT(self):
        label = pd.DataFrame()
        if len(self.df) > 1:
            label = self.df["TARGET"]

        return label


if __name__ == '__main__':
    pass

