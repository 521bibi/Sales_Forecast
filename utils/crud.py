import json

import numpy as np
import pandas as pd
from pymongo import MongoClient
import csv
import time

# 数据库连接
def connect_mongo(link_collection):
    mongo_uri = 'mongodb://localhost:27017/'
    client = MongoClient(mongo_uri)
    db = client.FIsalesForecast
    collection = db[link_collection]
    return collection


# 数据库连接
def connect_mongo_db(link_db, link_collection):
    mongo_uri = 'mongodb://localhost:27017/'
    client = MongoClient(mongo_uri)
    db = client[link_db]
    collection = db[link_collection]
    return collection


# 复制集合
def clone_db1todb2(adr1, db1, db1_collection):
    collection = db1 + '.' + db1_collection

    mongo_uri = 'mongodb://localhost:27017/'
    client = MongoClient(mongo_uri)
    db = client.FIsalesForecast
    db.cloneCollection(adr1, collection)

    return 0


# 新增 CSV2mongoDB
def insertcsv2MongoDB(file_csv, set1):
    with open(file_csv, 'r', encoding='utf-8') as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        csv_data = []
        # 创建一个counts计数一下 看自己一共添加了了多少条数据
        counts = 0
        index = 1
        for each in reader:
            csv_data.append(each)
            if index==10000:#10000个之后写入MongoDB中
                set1.insert_many(csv_data)
                csv_data.clear()
                index = 0
                print("成功添加了" + str(counts) + "条数据")
            counts+=1
            index+=1
        if len(csv_data)>0:#剩余的数据
            set1.insert_many(csv_data)
            print("成功添加了%s条数据"%len(csv_data))

# 新增 字典2mongoDB
async def dic2MongoDB(source_data, collection_in):
    try:
        data = source_data     #[dic]
        collection = collection_in      #存储位置
        collection.insert_many(data)
    except Exception as r:
        print('未知错误 %s' % r)

# 新增 pd.Dataframe新增到MongoDB
def DF2MongoDB(source_data, collection_in):
    try:
        DF = source_data     #[DF]
        collection = collection_in      #存储位置
        collection.insert_many(json.loads(DF.T.to_json()).values())
    except Exception as r:
        print('未知错误 %s' % r)

# 查询 集合全部查询
def findFromMongoDB(set1):
    mongo_find = list(set1.find())
    return mongo_find

# 查询 按条件查询 findquery条件、set2集合  查询结果return转为列表
def findFromMongoCondition(findquery, set2):
    myquery = findquery
    mydoc = set2.find(myquery)
    find_data = []
    # 创建一个counts计数一下 看自己一共查询了了多少条数据
    counts = 0
    for tmp in mydoc:
        find_data.append(tmp)
        counts += 1
    # print(find_data)
    print("成功查询了" + str(counts) + "条数据")
    return find_data

# 查询结果转化为dataframe
def find2pd(collection):
    cursor = collection.find()
    count = []
    for document in cursor:
        # print(list(document.keys()))
        count.append(document)
    dataframe = pd.DataFrame(count)
    # dataframe.to_csv('dataframe.csv')  # 保存CSV
    return dataframe


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S'))#计算时间用
    adr1 = 'mongodb://localhost:27017/'
    db1 = 'Datacenter'
    db1_collection = '12_744524_datain'
    adr2 = 'mongodb://localhost:27017/'
    db2 = 'FIsalesForecast'
    clone_db1todb2(adr1, db1, db1_collection)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
