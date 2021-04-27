from pymongo import MongoClient
import csv
import time
def connect_mongo():
    mongo_uri = 'mongodb://localhost:27017/'
    client = MongoClient(mongo_uri)
    db = client.FIdatabase
    collection = db['test']
    return collection

def insertToMongoDB(set1):
    with open('./trainData/12_141691_trainData.csv','r',encoding='utf-8') as csvfile:
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

def findFromMongoDB(set1):
    a = list(set1.find())
    return a

if __name__=='__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S'))#计算时间用
    set1 = connect_mongo()
    # insertToMongoDB(set1)
    print(findFromMongoDB(set1))
    print(time.strftime('%Y-%m-%d %H:%M:%S'))


