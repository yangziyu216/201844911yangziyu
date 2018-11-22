# -*- coding: utf-8 -*-
import os
import collections
import math

trainset_path='F:/data/data_train' #训练集数据路径
testset_path='F:/data/data_test' #测试集数据路径

def count_files(trainset_path):
    count=0
    floderlist=os.listdir(trainset_path)
    for floder in floderlist:
        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        count=count+len(filelist)
    return count

def pretreat(input):#数据预处理
    raw_data = []
    new_data=[]
    sort = []
    num = 0
    j=1
    for file1 in os.listdir(input):
        path1 = os.path.join(input, file1)
        num += 1
        for file2 in os.listdir(path1):
            path2 = os.path.join(path1, file2)
            sort.append(num)
            with open(path2, encoding='latin-1') as file:
                document = file.read()
                raw_data.append(document)
    # 预处理数据    
    for doc in raw_data:
        delpunctuation = re.compile('[%s]' % re.escape(string.punctuation))
        doc = delpunctuation.sub("", doc)
        words = []
        for word in doc:
            new_data.append(words)
    return len(new_data)

def get_trainVectors():

    totalfiles = count_files(trainset_path)
    floderlist = os.listdir(trainset_path)

    vectorlists =[] #存放整个训练集所有向量的列表

    j = 0 
    for floder in floderlist:

        j = j+1

        vector=[] #一个向量
        vector.append(floder) #追加类名
        totalwords = 0 #为每个类的所有单词计数
        filecount = 0.0 #统计该类中的文档数目
        wordsdict = {}

        floderpath=trainset_path+os.path.sep+floder
        filelist=os.listdir(floderpath)
        for file in filelist:
            filecount += 1
            filepath = floderpath+os.path.sep+file
            lines = open(filepath,'rb').readlines() #len(lines)即为该文档中的单词总数
            totalwords += len(lines)
            wordcountdic = collections.Counter(lines) #词典中为文档中的所有词，以及在该文档中出现的总次数
            for key, value in wordcountdic.items():
                key = key.strip()
                wordsdict[key] = wordsdict.get(key, 0)+value
        vector.append(totalwords)
        p = filecount / totalfiles
        vector.append(p)
        vector.append(wordsdict)

        vectorlists.append(vector)
    return vectorlists

# lists=get_trainVectors()


'''将测试集的每一个文档表示成向量，[类名，以该文档所有单词为元素的列表]。'''
def get_testVectors():

    totalfiles = count_files(testset_path)


    testVectors=[] #存放测试集所有向量的列表
    i=0 #统计向量个数

    floderlist=os.listdir(testset_path)
    for floder in floderlist:
        floderpath = testset_path + os.path.sep +floder
        filelist=os.listdir(floderpath)
        for file in filelist:

            i+=1

            vector=[] #一个文档表示成一个向量
            vector.append(floder) #追加类名
            words=[] #存放一个文档内所有的单词

            filepath=floderpath + os.path.sep +file
            lines=open(filepath,'rb').readlines()
            wordcountdic=collections.Counter(lines)
            for key,value in wordcountdic.items():
                key=key.strip()
                words.append(key)
            vector.append(words)
            testVectors.append(vector)
    return testVectors









def NB1():
    print('strat get vectors:)')
    trainVectorList=get_trainVectors()

    testVectorsList=get_testVectors()

    print('finish')

    allwords=pretreat(trainset_path) #allwords为数据集内单词总数
    success = 0
    failure=0 

    count=0 

    for i in range(len(testVectorsList)): #待分类的每一个文档

        count+=1

        eachclassp = []  # 存放一个文档在每个类别中的后验概率

        for j in range(len(trainVectorList)): #每一个类别

            p = trainVectorList[j][2] #概率初值设为该类别出现的概率
            p = math.log10(p)

            for word in testVectorsList[i][1]:
                numerator = float(trainVectorList[j][3].get(word, 0)+1)
                denominator = float(trainVectorList[j][1]+allwords)
                wordp = math.log10(numerator/denominator)

                p += wordp

            eachclassp.append((trainVectorList[j][0],p))

        eachclassp.sort(key=lambda x:x[1],reverse=True)

        judgeclass = eachclassp[0][0]

        if judgeclass == testVectorsList[i][0]:
            success = success+1
        else:
            failure = failure +1
    print ('测试集文档总数：' + str(len(testVectorsList)))

    successrate = float(success)/len(testVectorsList)
    print ('Accuracy：'+str(successrate))

NB1()





