# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:33:28 2023

@author: Juruoer
""" 
import numpy as np
from sophon.tools import somath
from sophon.tools import losses as los
from sophon.tools import miniBatch
from . import optimizer as opt
from . import layers
import math


class sequence:
    """
    外部可调用，初始化神经网络

    Parameters
    ----------
        
    layerList : list[layer]
        神经网络层列表，请至少包含一个层

    Returns
    -------
    一个神经网络实例

    """
    def __init__(self, inputShape:tuple, layerList:list, seed = 1):
        np.random.seed(seed)
        self.inputShape = (None, ) + inputShape
        
        # 检测层是否符合要求
        if (not isinstance(layerList, list)) or (len(layerList) == 0):
            raise TypeError("请输入一个包含至少一个 layer 的列表！")
        for lay in layerList:
            if not isinstance(lay, layers.layer):
                raise TypeError(f"列表中的元素类型必须是 layer ，但 {lay} 的类型为：{type(lay)} ！")
        self.layerList = layerList
        
        # 初始化各层参数
        inputShape = self.inputShape
        for lay in self.layerList:
            inputShape = lay.initParam(inputShape)
            
        # 其他参数
        self.isSoftmax = (self.layerList[-1].activationName == "softmax") # 输出是否为 somtmax
        self.outputSize = self.layerList[-1].unitSize # 每个示例的输出大小
        self.isOptions = False;
    
    
    def options(self, lr:float = 0.01, losses:str=None, optimizer = None):
        """
        神经网络选项

        Parameters
        ----------
        lr : float, optional
            学习率。默认为 0.01。
        losses : str, optional
            损失函数。 默认为 None，会与最后一层所选的激活函数对应。
        optimizer : TYPE, optional
            优化器。 默认为 None，即普通的梯度下降。

        Returns
        -------
        无。

        """
        if lr <= 0.:
            raise ValueError("学习率：lr 应大于 0！")
        self.lr = lr # 学习率
        
        if self.isSoftmax or (losses == None): # 当选择 softmax 时暂时不支持修改 losses
            self.losses, self.dlosses = los.normalLosses(self.layerList[-1].activationName) # 损失函数及其导数
        else:
            self.losses, self.dlosses = los.selectLosses(losses)
            
        if optimizer == None:
            self.optimizer = opt.normal()
        elif isinstance(optimizer, opt.normal):
            self.optimizer = optimizer
        else:
            raise TypeError(f"optimizer 必须为 sophon.neuralNetwork.optimizer.normal 类或其子类的实例，但 {optimizer} 的类型为：{type(optimizer)}！")
            
        self.isOptions = True
            
    
    def showOutputShape(self):
        """
        显示各层的输出的形状。

        Returns
        -------
        无。

        """
        print("\n")
        for i in range(len(self.layerList)):
            print(f"第 {i + 1} 层的输出形状：{self.layerList[i].outPutShape}")
    
    
    def prediction(self, input_x:np.ndarray):
        """
        对给定的输入进行预测

        Parameters
        ----------
        input_x : np.ndarray
            需要预测的输入。

        Returns
        -------
        input_x : TYPE
            预测结果。

        """
        for lay in self.layerList:
             input_x = lay.forward(input_x)[0]
        return input_x
    
    
    def fit(self, input_x:np.ndarray, input_y:np.ndarray, epochs:int = 200, batchSize:int=0, isRecord:bool = True, recordCount:int = 10, lrDecay:float=1.):
        """
        拟合。通过给定的输入和标签对网络进行训练。

        Parameters
        ----------
        input_x : np.ndarray
            输入
            
        input_y : np.ndarray
            标签
            
        epochs : int, optional
            遍历输入的次数，默认为 200
            
        batchSize : int, optional
            不为 0 表示启用 minibatch，且值表示 batchSize，为 0 表示不启用 minibatch，默认为 0。
            
        isRecord : bool, optional
            是否记录成本。默认为 True。
            
        recordCount : int, optional
            需要记录的成本的次数。默认为 10。
            
        lrDecay : float, optional
            学习率衰减参数，以控制学习率随着迭代次数而减少，该参数越小，衰减越快，默认为 1.

        Returns
        -------
        record : TYPE
            DESCRIPTION.

        """
        if self.isOptions == False:
            self.options()
            
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("迭代次数：epochs 应为大于 0 的整数")
        recordCount = int(somath.clamp(recordCount, 1, min(100000, epochs)))
        record = []
        
        batchSize = int(batchSize)
        if batchSize < 0:
            raise ValueError("小批量梯度下降所给定的没批大小：batchSize 应为大于等于 0 的整数！（当其为 0 时表示不启用该算法）")
            
        lrDecay = float(lrDecay)
        if lrDecay > 1. or lrDecay <= 0:
            raise ValueError("学习率衰减参数的范围应该在 (0, 1] 之间！")
        
        lr = self.lr
        
        m = input_x.shape[0]
        if input_x.shape[1:] != self.inputShape[1:]:
            raise ValueError("输入：input_x 形状不符合条件！")
        if input_y.shape != (m, self.outputSize):
            raise ValueError("期望结果：input_y 形状不符合条件！")
            
        if batchSize == 0: # 不启用 mini-batch
            batchs = [[input_x, input_y]]
        else:
            seed = 0 # 初始化随机种子
            
        t = 0 # 初始化迭代次数
            
        temp = math.ceil(epochs / recordCount);
        print("")
        for i in range(epochs): # 遍历 epoch 次
            print(f"epoch: {i + 1} / {epochs}")
            print("[" + "." * 20 + "]0%", end='')
            
            cost = 0. # 初始化成本
            if batchSize != 0:# 启用 mini-batch
                seed = seed + 1 # 随机种子递增，使得每次打乱都不同
                batchs = miniBatch.random_mini_batches(input_x, input_y, batchSize, seed)
                
            for j in range(len(batchs)): # 遍历所有批次
                t += 1 # 迭代次数更新，小批量迭代中，每遍历一个批次就相当于迭代一次
                
                preA = batchs[j][0]
                y = batchs[j][1]
                
                cacheStack = []
                
                for lay in self.layerList: # 前向传播
                    preA, cache = lay.forward(preA) # 用于下一层 forward 的 preA 和用于本层 backward 的 cache
                    cacheStack.append(cache)
                cost += self.losses(y, preA)
                    
                if self.isSoftmax:
                    da = np.subtract(preA, y) # softmax 的 losses 固定，且放在最后一层，此处用 dz 代替 da
                else:
                    da = self.dlosses(y, preA)
                    
                k = len(self.layerList) - 1
                
                while k > -1: # 反向传播
                    lay = self.layerList[k]
                    if lay.needUpdaParam: # 需要更新参数
                        da, dw, db = lay.backward(da, cacheStack[-1])
                        dw, db, lay.optimizerParam = self.optimizer.cel(t, dw, db, lay.optimizerParam) # 优化器
                        lay.weight = np.subtract(lay.weight, np.multiply(lr, dw)) # 更新本层权重
                        lay.bias = np.subtract(lay.bias, np.multiply(lr, db)) # 更新本层偏移量
                    else: # 不需要更新参数
                        da = lay.backward(da, cacheStack[-1])
                    cacheStack.pop()
                    k -= 1
                
                lr = lr * lrDecay # 学习率衰减
                progress = (j + 1) / len(batchs)
                cnt = int(progress * 20)
                print("\r[" + "=" * cnt + "." * (20 - cnt) + "]" + str(int(progress * 100)) + "%", end='')
            
            cost = cost / m
            print(f" 成本：{cost}\n")
            if isRecord and (i % temp == 0): # 记录迭代成本
                record.append((i, cost))
        return record