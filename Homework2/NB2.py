# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 19:39:45 2018

@author: Administrator
"""

from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer 
from sklearn import metrics 
from sklearn.naive_bayes import BernoulliNB

'''使用python的机器学习库，使用sklearn自带的贝叶斯分类器完成文本分类
类别太多没有写一个函数 直接读取并添加名称 选取十个文件夹操作'''



def get_dataset(): #获取地址和文件夹的标签以上的代码就是读取全部数据，包括训练集和测试集，并随机打乱，
    data=[]
    for root, dirs, files in os.walk(r'F:\data\20news-18828\alt.atheism'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'alt.atheism')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\comp.graphics'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'comp.graphics')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\comp.os.ms-windows.misc'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'comp.os.ms-windows.misc')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\comp.sys.ibm.pc.hardware'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'comp.sys.ibm.pc.hardware')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\comp.sys.mac.hardware'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'comp.sys.mac.hardware')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\comp.windows.x'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'comp.windows.x')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\misc.forsale'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'misc.forsale')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\rec.autos'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'rec.autos')) 
    for root, dirs, files in os.walk(r'F:\data\20news-18828\rec.motorcycles'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'rec.motorcycles'))
    for root, dirs, files in os.walk(r'F:\data\20news-18828\rec.sport.baseball'): 
        for file in files: 
            realpath = os.path.join(root, file) 
            with open(realpath, errors='ignore') as f: 
                data.append((f.read(), 'rec.sport.baseball'))
    random.shuffle(data)
    return data

    
    
    
    
data=get_dataset()#读取数据

def train_and_test_data(data_): #划分训练集和测试集
    filesize = int(0.8 * len(data_)) 
    # 训练集和测试集的比例为8:2 
    train_data_ = [each[0] for each in data_[:filesize]] 
    train_target_ = [each[1] for each in data_[:filesize]] 
    test_data_ = [each[0] for each in data_[filesize:]] 
    test_target_ = [each[1] for each in data_[filesize:]] 
    return train_data_, train_target_, test_data_, test_target_



train_data, train_target, test_data, test_target = train_and_test_data(data)
print('训练集数量：',len(train_data))
print('测试集数量：',len(test_data))
nbc = Pipeline([ ('vect', TfidfVectorizer( )), ('clf', MultinomialNB(alpha=1.0)), ]) #MultinomialNB假设特征的先验概率为多项式分布
nbc.fit(train_data, train_target) #训练多项式模型贝叶斯分类器 
predict = nbc.predict(test_data) #在测试集上预测结果 
count = 0 #统计预测正确的结果个数 
#print(predict)
#print(test_target)
for left , right in zip(predict, test_target): 
    if left == right: 
        count += 1 
print('Accuracy')
print(count/len(test_target))
#print(nbc)

