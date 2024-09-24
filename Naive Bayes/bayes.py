from functools import reduce

import numpy as np
from numpy import matrix


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]               #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec


# according to the postingList, split it to vector
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec

# disposal sample data, return the sole word
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union
    return list(vocabSet)


# naive bayes training function
def trainNB0(trainMatrix, trainCategory):
    # calculate the number of document
    numTrainDocs = len(trainMatrix)
    # calculate the words of a document
    numWords = len(trainMatrix[0])
    # the prob of abusive
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log (p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0



def testingNB():
    listOPosts, listClasses = loadDataSet()  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    print(f"p0v: {p0V}")
    print(f"p1v: {p1V}")
    testEntry = ['love', 'my', 'dalmation']  # 测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')  # 执行分类并打印分类结果
    testEntry = ['stupid', 'garbage']  # 测试样本2

    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))  # 测试样本向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')


if __name__ == '__main__':
    testingNB()