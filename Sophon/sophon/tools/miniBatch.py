# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:18:07 2023

@author: Juruoer
"""
import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    从 (X，Y) 创建随机小批次列表

    Parameters
    ----------
    X : TYPE
        输入的示例
    Y : TYPE
        标签值
    mini_batch_size : TYPE, optional
        小批量的大小，整数。
        
    seed : TYPE, optional
        随机种子，默认为 0。

    Returns
    -------
    mini_batches : TYPE
        从 (X，Y) 中创建的随机小批次列表

    """    
    np.random.seed(seed)
    m = X.shape[0]
    mini_batches = []
        
    # 打乱 (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, -1))

    # 生成 mini-batchs
    num_complete_minibatches = math.floor(m/mini_batch_size) # 完整含有 mini_batch_size 大小的批次
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size, :]
        
        mini_batch = [mini_batch_X, mini_batch_Y]
        mini_batches.append(mini_batch)
    
    # 可以还有剩余部分
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[int(m/mini_batch_size)*mini_batch_size : , :]
        mini_batch_Y = shuffled_Y[int(m/mini_batch_size)*mini_batch_size : , :]
        
        mini_batch = [mini_batch_X, mini_batch_Y]
        mini_batches.append(mini_batch)
    
    return mini_batches