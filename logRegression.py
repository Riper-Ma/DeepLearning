###################################
# Regression:Logistic Regression
# Author:RiperMa
# Date:2017.9.30
###################################


from numpy import *
import matplotlib.pyplot as plt
import time

#calculate the sigmod function
def sigmod(inX):
    return 1.0 / (1+ exp(-inX))

def loadDataSet():
    dataMartix = []
    dataLablel = []

    f = open('testSet.txt')
    for line in f.readlines():
        #print(line)
        lineList = line.strip().split()
        dataMartix.append([1, float(lineList[0]), float(lineList[1])])
        dataLablel.append(int(lineList[2]))
    #for i in range(len(dataMartix)):
    #   print(dataMartix[i])
    #print(dataLablel)
    #print(mat(dataLablel).transpose())
    matLabel = mat(dataLablel).transpose()
    return dataMartix, matLabel

def graAscent(dataMartix, matLabel):
    m, n = shape(dataMartix)
    matMartix = mat(dataMartix)

    w = ones((n,1))
    alpha = 0.001
    num = 500
    for i in range(num):
        error = sigmod(matMartix * w) - matLabel
        w = w - alpha * matMartix.transpose() * error
    return w

def stocGraAscent(dataMartix, matLabel):
    m, n = shape(dataMartix)
    matMartix = mat(dataMartix)

    w = ones((n, 1))
    alpha = 0.001
    num = 50
    for i in range(num):
        for j in range(m):
            error = sigmod(matMartix[j] * w) - matLabel[j]
            w = w - alpha * matMartix[j].transpose() * error
    return w


def stocGraAscent1(dataMartix, matLabel):
    m, n = shape(dataMartix)
    matMartix = mat(dataMartix)

    w = ones((n, 1))
    num = 500
    setIndex = set([])
    for i in range(m):
        for j in range(m):
            alpha = 4 / (1 + i + j) + 0.001

            dataIndex = random.randint(0, 100)
            while dataIndex in setIndex:
                setIndex.add(dataIndex)
                dataIndex = random.randint(0,100)
            error = sigmod(matMartix[dataIndex] * w) - matLabel[dataIndex]
            w = w - alpha * matMartix[dataIndex].transpose() * error
    return w


def draw(weight):
    x0List = []; y0List = [];
    x1List = []; y1List = [];
    f = open('testSet.txt', 'r')
    for line in f.readlines():
        lineList=line.strip().split()
        if lineList[2]=='0':
            x0List.append(float(lineList[0]))
            y0List.append(float(lineList[1]))
        else:
            x1List.append(float(lineList[0]))
            y1List.append(float(lineList[1]))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(x0List,y0List,s=10,c='red')
    ax.scatter(x1List,y1List,s=10,c='green')

    xList=[];yList=[]
    x=arange(-3,3,0.1)
    for i in arange(len(x)):
        xList.append(x[i])

    y=(-weight[0]-weight[1]*x)/weight[2]
    for j in arange(y.shape[1]):
        yList.append(y[0,j])

    ax.plot(xList,yList)
    plt.xlabel('x1');plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    dataMatrix,matLabel=loadDataSet()
    #weight=graAscent(dataMatrix,matLabel)
    weight=stocGraAscent1(dataMatrix,matLabel)
    print(weight)
    draw(weight)
