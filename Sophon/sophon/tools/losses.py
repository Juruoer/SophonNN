# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:51:48 2023

@author: Juruoer
"""
import numpy as np

def squaredErrorLoss(input_y, pre_y)->float:
    return np.sum(np.power(np.subtract(pre_y, input_y), 2)) / 2


def dsquaredErrorLoss(input_y, pre_y):
    return np.subtract(pre_y, input_y)


def binaryCrossentropy(input_y, pre_y)->float:
    return -np.sum(np.add(np.multiply(input_y, np.log(pre_y + 1e-8)), np.multiply((1 - input_y), np.log(1 - pre_y + 1e-8))))


def dbinaryCrossentropy(input_y, pre_y):
    return np.subtract(np.divide((1 - input_y), (1 - pre_y + 1e-8)), np.divide(input_y, (pre_y + 1e-8)))


def sparseCategoricalCrossentropy(input_y, pre_y)->float:
    return -np.sum(np.multiply(input_y, np.log(pre_y + 1e-6)))


def dsparseCategoricalCrossentropy(input_y, pre_y):
    return -np.divide(input_y, (pre_y + 1e-6))



lossesList = { 
        "squaredErrorLoss" : (squaredErrorLoss, dsquaredErrorLoss), 
        "binaryCrossentropy" : (binaryCrossentropy, dbinaryCrossentropy), 
        "sparseCategoricalCrossentropy" : (sparseCategoricalCrossentropy, dsparseCategoricalCrossentropy)
    }


normalLossList = {"linear" : "squaredErrorLoss", "relu" : "squaredErrorLoss", "sigmoid" : "binaryCrossentropy", "softmax" : "sparseCategoricalCrossentropy"}


def selectLosses(lossesName:str):
    if lossesName in lossesList:
        return lossesList[lossesName]
    raise ValueError(f"损失函数：{lossesName} 不存在！")
    
    
def normalLosses(activationName:str):
    if activationName in normalLossList:
        return lossesList[normalLossList[activationName]]
    raise ValueError(f"激活函数：{activationName} 不存在！")