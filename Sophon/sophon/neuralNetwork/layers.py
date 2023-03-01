# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 21:11:30 2023

@author: Juruoer
"""
import numpy as np
from sophon.tools import activation as act


class layer:
    def __init__(self, needUpdaParam:bool):
        self.needUpdaParam = needUpdaParam
        self.outPutShape = None


class dense(layer):
    """
    致密层

    Parameters
    ----------
    unitSize : int
        神经元数量
        
    activation : str
        激活函数


    Returns
    -------
    一个致密层

    """
    def __init__(self, unitSize:int, activation:str):
        unitSize = int(unitSize)
        if unitSize < 1:
            raise ValueError("神经元数量：unitSize 应为大于 0 的整数！")
        self.unitSize = unitSize # 神经元数量
        self.activation, self.dactivation = act.selectActivation(activation) # 激活函数及其导函数
        self.activationName = activation # 激活函数的名称
        self.weightInitParam = 2 if self.activationName == "relu" else 1 # 权重矩阵初始化时的参数
        self.weight = None # 权重矩阵
        self.bias = None # 偏移量矩阵
        self.optimizerParam = None # optimizer 参数
        super().__init__(True)
        
        
    def initParam(self, inputShape):
        n = inputShape[1]
        self.weight = np.multiply(np.random.randn(n, self.unitSize), np.sqrt(self.weightInitParam / n))
        self.bias = np.zeros((1, self.unitSize))
        self.outPutShape = (None, self.unitSize)
        return self.outPutShape
    
    
    def forward(self, preA:np.ndarray):
        z = np.add(np.dot(preA, self.weight), self.bias)
        return self.activation(z), (preA, z)
    
    
    def backward(self, da:np.ndarray, cache:tuple):
        preA, z = cache
        if self.activationName == "softmax": # 对 softmax 的特殊处理
            dz = da
        else:
            dz = np.multiply(da, self.dactivation(z));
        k = 1. / preA.shape[0]
        dw = np.multiply(k, np.dot(preA.T, dz));
        db = np.multiply(k, np.sum(dz, axis=0, keepdims=True))
        preDa = np.dot(dz, self.weight.T)
        return preDa, dw, db
    

class relu(layer):
    """
    relu 层，将输入的特征中小于 0 的值设置为 0

    Returns
    -------
    一个 relu 层

    """
    def __init__(self):
        super().__init__(False)
    
    
    def initParam(self, inputShape):
        self.outPutShape = inputShape
        return self.outPutShape
    
    
    def forward(self, preA:np.ndarray):
        return act.relu(preA), (preA, )
    
    
    def backward(self,  da:np.ndarray, cache:tuple):
        return np.multiply(da, act.drelu(cache[0]))
        

class conv(layer):
    """
    卷积层

    Parameters
    ----------
    filters : TYPE
        卷积核个数。
    kernel_size : TYPE, optional
        卷积核大小。默认为 3。
    pad : TYPE, optional
        填充大小。 默认为 0。
    stride : TYPE, optional
        卷积步长。 默认为 1。

    Returns
    -------
    一个卷积层

    """
    def __init__(self, filters, kernel_size=3, pad=0, stride=1):
        filters = int(filters)
        self.filters = filters # 卷积核个数
        kernel_size = int(kernel_size)
        self.kernel_size = kernel_size # 卷积核大小
        pad = int(pad)
        self.pad = pad # 填充宽度
        stride = int(stride)
        if stride < 1:
            raise ValueError("卷积步长必须为大于 0 的整数！")
        self.stride = stride # 步长
        self.weightInitParam = 1
        self.weight = None # 权重矩阵
        self.bias = None # 偏移量矩阵
        self.optimizerParam = None # optimizer 参数
        super().__init__(True)
        
        
    def initParam(self, inputShape):
        n_H_prev, n_W_prev, n_C_prev = inputShape[1:]
        
        temp = 2 * self.pad - self.kernel_size
        n_H = int((n_H_prev + temp)/self.stride) + 1 # 当前层输出的高度
        n_W = int((n_W_prev + temp)/self.stride) + 1 # 当前层输出的宽度
        
        n = n_H_prev * n_W_prev * n_C_prev
        self.weight = np.random.randn(self.kernel_size, self.kernel_size, n_C_prev, self.filters) * np.sqrt(self.weightInitParam / n)
        self.bias = np.zeros((1, 1, 1, self.filters))
        self.outPutShape = (None, n_H, n_W, self.filters)
        return self.outPutShape
        
    
    
    def zero_pad(self, x):
        if self.pad == 0:
            return x
        return np.pad(x, ((0,0), (self.pad, self.pad), (self.pad, self.pad), (0,0)))    
    
    
    def conv_single_step(self, a_slice_prev, W, b):
        s = np.multiply(a_slice_prev, W)
        return np.add(np.sum(s), np.squeeze(b))
    
        
    def forward(self, preA:np.ndarray):
        (m, n_H_prev, n_W_prev, n_C_prev) = preA.shape
        
        n_H, n_W = self.outPutShape[1:3]
        
        a = np.zeros((m, n_H, n_W, self.filters)) # 初始化输出
        
        # 将输入进行填充
        preA_pad = self.zero_pad(preA)
        for i in range(m):
            preA_pad_i = preA_pad[i]          # 当前批次的第 i 个输入
            for h in range(n_H):           # 计算输出的第 h 行
                vert_start = self.stride * h # 卷积起始行
                vert_end = vert_start  + self.kernel_size # 卷积结束行
                
                for w in range(n_W):       # 计算输出的第 w 列
                    horiz_start = self.stride * w # 卷积起始列
                    horiz_end = horiz_start + self.kernel_size # 卷积结束列
                    
                    for c in range(self.filters):   # 输出的第 c 层
                        preA_slice_i = preA_pad_i[vert_start:vert_end,horiz_start:horiz_end,:] # 从输入中取出当前用于当前卷积的部分
                        
                        # 卷积核中用于当前卷积计算的部分
                        weights = self.weight[:, :, :, c]
                        biases  = self.bias[:, :, :, c]
                        a[i, h, w, c] = self.conv_single_step(preA_slice_i, weights, biases) # 卷积
        
        return a, (preA, )
    
    
    def backward(self, da:np.ndarray, cache:tuple):
        preA = cache[0]
        (m, n_H_prev, n_W_prev, n_C_prev) = preA.shape
        
        n_H, n_W = self.outPutShape[1:3]
        
        preDa = np.zeros(preA.shape) #  需要计算的前一层的 da                 
        dw = np.zeros(self.weight.shape) # 本层的 dw
        db = np.zeros(self.bias.shape) # 本层的 db
        
        preA_pad = self.zero_pad(preA)
        preDa_pad = self.zero_pad(preDa)
        
        for i in range(m):
            preA_pad_i = preA_pad[i]
            preDa_pad_i = preDa_pad[i]
            
            for h in range(n_H):
                vert_start = self.stride * h 
                vert_end = vert_start + self.kernel_size
                for w in range(n_W):
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + self.kernel_size
                    for c in range(self.filters):
                        # 当前卷积对应的输入部分
                        preA_slice_i = preA_pad_i[vert_start:vert_end,horiz_start:horiz_end,:]
                        # 计算梯度
                        preDa_pad_i[vert_start:vert_end, horiz_start:horiz_end, :] += np.multiply(self.weight[:,:,:,c], da[i, h, w, c])
                        dw[:,:,:,c] += np.multiply(preA_slice_i, da[i, h, w, c])
                        db[:,:,:,c] += da[i, h, w, c]
                        
            if self.pad == 0:
                preDa[i, :, :, :] = preDa_pad_i[:, :, :]
            else:
                preDa[i, :, :, :] = preDa_pad_i[self.pad:-self.pad, self.pad:-self.pad, :]
        
        # Making sure your output shape is correct
        assert(preDa.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return preDa, dw, db


class maxPool(layer):
    """
    max 池化层

    Parameters
    ----------
    kernel_size : TYPE, optional
        池化核大小。默认为 2。
    stride : TYPE, optional
        池化步长。默认为 2。

    Returns
    -------
    一个 max 池化层
    """
    def __init__(self, kernel_size=2, stride=2):
        kernel_size = int(kernel_size)
        self.kernel_size = kernel_size # 池化核大小
        stride = int(stride)
        self.stride = stride # 步幅
        super().__init__(False)
        
    
    def initParam(self, inputShape):
        n_H_prev, n_W_prev, n_C = inputShape[1:]
        
        n_H = int((n_H_prev - self.kernel_size) / self.stride) + 1
        n_W = int((n_W_prev - self.kernel_size) / self.stride) + 1
        
        self.outPutShape = (None, n_H, n_W, n_C)
        return self.outPutShape
        
    
    def create_mask_from_window(self, x):
        return (x == np.max(x))
    
    
    def forward(self, preA:np.ndarray):
        (m, n_H_prev, n_W_prev, n_C_prev) = preA.shape
        n_H, n_W, n_C = self.outPutShape[1:]
        a = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):
            preA_i = preA[i]
            for h in range(n_H):
                vert_start = self.stride * h 
                vert_end = vert_start + self.kernel_size
                
                for w in range(n_W):
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + self.kernel_size
                    
                    for c in range (n_C):
                        # 输入中当前池化的部分
                        preA_slice_i = preA_i[vert_start:vert_end,horiz_start:horiz_end,c]
                        a[i, h, w, c] = np.max(preA_slice_i) 
        
        return a, (preA, )
    
    
    def backward(self, da, cache):
        preA = cache[0]
        m, n_H_prev, n_W_prev, n_C_prev = preA.shape
        n_H, n_W, n_C = self.outPutShape[1:]
        
        preDa = np.zeros(preA.shape)
        
        for i in range(m):
            preA_i = preA[i]
            
            for h in range(n_H):
                vert_start  = h * self.stride
                vert_end    = h * self.stride + self.kernel_size
                for w in range(n_W):
                    horiz_start = w * self.stride
                    horiz_end   = w * self.stride + self.kernel_size
                    for c in range(n_C):
                        preA_slice_i = preA_i[vert_start:vert_end, horiz_start:horiz_end, c]    
                        mask = self.create_mask_from_window(preA_slice_i) #  获取 mask，即标记真正对成本有效的特征
                        preDa[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, da[i, h, w, c])
        
        return preDa


class avgPool(layer):
    """
    avg 池化层

    Parameters
    ----------
    kernel_size : TYPE, optional
        池化核大小。默认为 2。
    stride : TYPE, optional
        池化步长。默认为 2。

    Returns
    -------
    一个 avg 池化层
    """
    def __init__(self, kernel_size=2, stride=2):
        kernel_size = int(kernel_size)
        self.kernel_size = kernel_size # 池化核大小
        self.avg_reciprocal = 1. / (kernel_size**2) # 每个特征对成本的贡献比例
        stride = int(stride)
        self.stride = stride # 步幅
        super().__init__(False)
    
    
    def initParam(self, inputShape):
        n_H_prev, n_W_prev, n_C = inputShape[1:]
        
        n_H = int((n_H_prev - self.kernel_size) / self.stride) + 1
        n_W = int((n_W_prev - self.kernel_size) / self.stride) + 1
        
        self.outPutShape = (None, n_H, n_W, n_C)
        return self.outPutShape
        
        
    def distribute_value(self, dz):
        return np.multiply(self.avg_reciprocal, np.multiply(dz, np.ones((self.kernel_size, self.kernel_size))))
    
    
    def forward(self, preA:np.ndarray):
        (m, n_H_prev, n_W_prev, n_C_prev) = preA.shape
        n_H, n_W, n_C = self.outPutShape[1:]
        a = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):
            preA_i = preA[i]
            for h in range(n_H):
                vert_start = self.stride * h 
                vert_end = vert_start + self.kernel_size
                
                for w in range(n_W):
                    horiz_start = self.stride * w
                    horiz_end = horiz_start + self.kernel_size
                    
                    for c in range (n_C):
                        preA_slice_i = preA_i[vert_start:vert_end,horiz_start:horiz_end,c]
                        a[i, h, w, c] = np.mean(preA_slice_i) 
        
        return a, (preA, )
    
    
    def backward(self, da, cache):
        preA = cache[0]
        m, n_H_prev, n_W_prev, n_C_prev = preA.shape
        n_H, n_W, n_C = self.outPutShape[1:]
        preDa = np.zeros(preA.shape)
        
        for i in range(m):
            for h in range(n_H):
                vert_start  = h * self.stride
                vert_end    = h * self.stride + self.kernel_size
                for w in range(n_W):
                    horiz_start = w * self.stride
                    horiz_end   = w * self.stride + self.kernel_size
                    for c in range(n_C):
                        da_i = da[i, h, w, c]
                        preDa[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da_i)
        
        return preDa
    

class flatten(layer):
    def __init__(self):
        """
        扁平层，用于将卷积层的输出转为全连接层的输入

        Returns
        -------
        一个扁平层

        """
        super().__init__(False)
        
    
    def initParam(self, inputShape):
        n_H_prev, n_W_prev, n_C = inputShape[1:]
        n = n_H_prev * n_W_prev * n_C
        
        self.outPutShape = (None, n)
        return self.outPutShape
    
    
    def forward(self, preA:np.ndarray):
        preAShape = preA.shape
        return preA.reshape((preAShape[0], -1)), (preAShape, )
           
    
    def backward(self, da, cache):
        return da.reshape(cache[0])
    
