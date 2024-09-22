import operator

import numpy as np


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [155, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

def classify(inX, dataSet, labels, k):
    # numpy function shape[0] return the row of dataSet
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # return the index of sorted array
    sortedDistIndices = distances.argsort()
    # define a dict to record times
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        # 获取键voteIlabel的值默认为0, 并加一重新赋值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]





if __name__ == '__main__':
    group, labels = createDataSet()
    test = [100, 20]
    test_class = classify(test, group, labels, 1)
    print(test_class)