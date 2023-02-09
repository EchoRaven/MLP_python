import numpy as np
import random
import pandas as pd

def Sigmoid(x):
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


sigmoid = Sigmoid


class Network:
    #输入层大小
    inputSize = 2
    #输入层
    inputLayer = np.array([], dtype="float32")
    #输出层大小
    outputSize = 2
    #输出层
    outputLayer = np.array([], dtype="float32")
    #隐藏层
    hiddenLayer = []
    #轮数
    turn = 1000
    #步长
    step = 0.01
    #转换矩阵(不包括偏置项theta0)
    theta = []
    #偏执项(theta0)
    bias = []
    #误差
    diff = []
    #反向传递梯度
    delta = []
    #激活函数
    activeFunction = sigmoid
    #正则化系数
    p1 = 0
    #测试用例
    train_test = ""
    #输入数据
    input_data = []
    #输出数据
    output_data = []

    # 初始化各个参数
    def __init__(self,
                 inputSize = 2,
                 outputSize = 2,
                 hiddenStructure = [3, 4, 3],
                 turn = 1000,
                 step = 0.01,
                 randomInit = True,
                 activeFunction = sigmoid,
                 p1 = 0
                 ):
        self.inputLayer = np.zeros([inputSize, 1], dtype="float32")
        self.outputLayer = np.zeros([outputSize, 1], dtype="float32")
        for size in hiddenStructure:
            self.hiddenLayer.append(np.zeros([size, 1], dtype="float32"))
        #插入输入层到第一隐藏层的转换矩阵
        hiddenLen = len(hiddenStructure)
        self.theta.append(np.zeros([hiddenStructure[0], inputSize], dtype="float32"))
        self.delta.append(np.zeros([hiddenStructure[0], inputSize], dtype="float32"))
        for index in range(hiddenLen - 1):
            self.theta.append(
                np.zeros([hiddenStructure[index+1],
                hiddenStructure[index]],
                dtype="float32"))
            self.delta.append(
                np.zeros([hiddenStructure[index + 1],
                hiddenStructure[index]],
                dtype="float32"))
        self.theta.append(np.zeros([outputSize, hiddenStructure[hiddenLen-1]], dtype="float32"))
        self.delta.append(np.zeros([outputSize, hiddenStructure[hiddenLen - 1]], dtype="float32"))
        for index in range(hiddenLen):
            self.bias.append(np.zeros([hiddenStructure[index], 1], dtype="float32"))
            self.diff.append(np.zeros([hiddenStructure[index], 1], dtype="float32"))
        self.bias.append(np.zeros([outputSize, 1], dtype="float32"))
        self.diff.append(np.zeros([outputSize, 1], dtype="float32"))
        if randomInit:
            for layer in self.theta:
                for x in range(len(layer)):
                    for y in range(len(layer[x])):
                        layer[x][y] = random.uniform(-1, 1)
            for layer in self.bias:
                for index in range(len(layer)):
                    layer[index] = random.uniform(-1, 1)
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.turn = turn
        self.step = step
        self.activeFunction = activeFunction
        self.p1 = p1

    #前向传播
    def ForwardPropagation(self, inputData):
        self.inputLayer = inputData
        for index in range(len(self.theta)):
            if index == 0:
                self.hiddenLayer[index] = self.activeFunction(
                    np.matmul(self.theta[index], self.inputLayer)
                    +self.bias[index])
            elif index == len(self.theta)-1:
                self.outputLayer = self.activeFunction(
                    np.matmul(self.theta[index], self.hiddenLayer[index-1])
                    +self.bias[index])
            else:
                self.hiddenLayer[index] = self.activeFunction(
                    np.matmul(self.theta[index], self.hiddenLayer[index-1])
                    +self.bias[index])

    #计算梯度
    def CalDelta(self, inputData, result):
        self.ForwardPropagation(inputData)
        hiddenLen = len(self.hiddenLayer)
        self.diff[hiddenLen] = self.outputLayer - result
        #非偏执部分
        for index in range(hiddenLen-1, -1, -1):
            self.diff[index] = np.matmul(self.theta[index+1].T, self.diff[index+1])\
                               *(self.hiddenLayer[index]*(1-self.hiddenLayer[index]))
        for index in range(hiddenLen, -1, -1):
            if index!=0:
                self.delta[index] += np.matmul(self.diff[index], self.hiddenLayer[index-1].T)
            else:
                self.delta[index] += np.matmul(self.diff[index], self.inputLayer.T)
        var = 0
        for num in self.diff[hiddenLen]:
            var += num*num
        return var

    #反向传递
    def BackPropagation(self, inputData, result):
        #delta清零
        for layer in self.delta:
            for index in range(len(layer)):
                layer[index] = 0
        #累计Delta
        tot=0
        for index in range(len(result)):
            tot+=self.CalDelta(inputData[index], result[index])
        #计算总循环次数
        m = len(result)
        hiddenLen = len(self.hiddenLayer)
        for index in range(hiddenLen+1):
            self.theta[index]-=self.step*(self.delta[index]/m+self.p1/m*self.theta[index])
            self.bias[index]-=self.step*(self.diff[index]/m)
        print("误差 : ", tot/(2*m))

    def GetData(self, filename):
        self.train_test = filename
        data = pd.read_csv(filename, sep=";", header=None)
        for index in range(1, len(data.values)):
            l = len(data.values[index])
            self.input_data.append(np.array(data.values[index][0:l-1], dtype="float32").reshape([self.inputSize,1]))
            out = np.zeros([10,1], dtype="float32")
            out[int(data.values[index][l-1])-1]=1
            self.output_data.append(out)

    def Train(self, filename):
        self.GetData(filename)
        for t in range(self.turn):
            print("number : ", t)
            self.BackPropagation(self.input_data, self.output_data)

    #测试用例
    def ShowHiddenLayer(self):
        print(self.hiddenLayer)

    #显示转换举证
    def ShowTheta(self):
        print(self.bias)
        print(self.theta)

    #显示结果
    def ShowOutput(self):
        print(self.outputLayer)

    #显示梯度
    def ShowDelta(self):
        print(self.delta)

    #显示损失
    def ShowDiff(self):
        print(self.diff)