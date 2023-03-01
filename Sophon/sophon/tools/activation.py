# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:13:42 2023

@author: Juruoer
"""
import numpy as np

def linear(z:np.ndarray)->np.ndarray:
    return z


def dlinear(z:np.ndarray):
    return np.ones_like(z)


def relu(z:np.ndarray)->np.ndarray:
    return np.where(z > 0., z , 0.)
    

def drelu(z:np.ndarray):
    return np.where(z > 0., 1., 0.)


def sigmoid(z:np.ndarray)->np.ndarray:
    return np.reciprocal(np.add(1., np.exp(-z)))


def dsigmoid(z:np.ndarray):
    a = sigmoid(z)
    return np.multiply(np.subtract(1., a), a)


def softmax(z:np.ndarray)->np.ndarray:
    c = np.max(z, axis=1, keepdims=True)
    g = np.exp(np.subtract(z, c))
    s = np.sum(g, axis=1, keepdims=True)
    return np.divide(g, s)


def dsoftmax(z:np.ndarray)->np.ndarray:
    pass


def tanh(z:np.ndarray)->np.ndarray:
    t1 = np.exp(z)
    t2 = np.exp(-z)
    return np.divide(np.subtract(t1, t2), np.add(t1, t2))

def dtanh(z:np.ndarray)->np.ndarray:
    return np.subtract(1., np.power(tanh(z), 2))


activationList = {
        "linear" : (linear, dlinear), 
        "relu" : (relu, drelu), 
        "sigmoid" : (sigmoid, dsigmoid), 
        "softmax" : (softmax, dsoftmax)
    };


def selectActivation(activationName:str):
    if activationName in activationList:
        return activationList[activationName]
    raise ValueError(f"激活函数：{activationName} 不存在")