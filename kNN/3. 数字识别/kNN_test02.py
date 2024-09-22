import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN

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

def img2vector(filename):
    # create 1x1024 zero vector
    returnVect = np.zeros((1, 1024))  # 2d
    # open file
    fr = open(filename)
    # read in line
    for i in range(32):
        # read one line
        lineStr = fr.readline()
        # add the first 32 elements of each row to returnVect
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    # return the filename below the trainingDigits document
    trainingFileList = listdir('trainingDigits')
    # return the number of files
    m = len(trainingFileList)
    # init trainingMat, testMat
    trainingMat = np.zeros((m, 1024))
    # resolve the training set category from the filename
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # get category
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector(f'trainingDigits/{fileNameStr}')
    # create kNN classifier
    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    # fit model
    testFileList = listdir('testDigits')
    errorCount = 0.0
    # the number of test data
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector(f'testDigits/{fileNameStr}')
        # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print(f"result of classifier: {classifierResult},  the true result: {classNumber}")
        if(classifierResult != classNumber):
            errorCount == 1.0
    print(f"total error data number: {errorCount}, error ratio: {errorCount/mTest * 100}%")



if __name__ == '__main__':
    handwritingClassTest()