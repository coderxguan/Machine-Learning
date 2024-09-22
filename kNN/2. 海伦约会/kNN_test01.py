import operator

import matplotlib.pyplot as plt
import numpy as np
import  sklearn

# 打开并解析文件, 对数据进行分类,
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # get the rows
    numberOfLines = len(arrayOLines)
    # 创建以文件行数为行, 3个特征值的矩阵
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # s.strip(rm) rm为空, 默认删除首尾空白符(\n, \r, \t ' ')
        line = line. strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

# 数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) # sum row
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    filename = 'datingTestSet.txt'
    datingDateMat, datingLabels = file2matrix(filename)
    hoRatio = 0.10
    normMat, ranges, minvals = autoNorm(datingDateMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))



# 通过输入一个人的三维特征, 进行分类输出
def classifyPerson():
    resultList = ['hate', 'a little like', 'very like']
    percentTats = float(input("The percent of play video game:"))
    ffMiles = float(input("Fly miles annually:"))
    iceCream = float(input("ice Cream every week:"))
    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    normInArr = (inArr -  minVals) / ranges
    classifyResult = classify0(normInArr, normMat, datingLabels, 3)
    print(f"you could {resultList[classifyResult-1]} this person")


if __name__ == '__main__':
    # datingClassTest()
    # classifyPerson()
    # print(np.__version__)
    # print(np.show_config())
    print(sklearn.__version__)























