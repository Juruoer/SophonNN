# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:59:11 2023

@author: Juruoer
"""
import numpy as np

class normal():
    def __init__(self):
        pass
    
    
    def cel(self, t, dw, db, param=None):
        return dw, db, param


class momentum(normal):
    def __init__(self, beta:float = 0.9):
        self.beta = beta
        self.rbeta = 1. - self.beta
    
    
    def cel(self, t, dw, db, param=None):
        # param = [Vdw, Vdb]
        if param == None:
            param = [0, 0]
        param[0] = np.add(np.multiply(self.beta, param[0]), np.multiply(self.rbeta, dw))
        param[1] = np.add(np.multiply(self.beta, param[1]), np.multiply(self.rbeta, db))
        dw = param[0]
        db = param[1]
        return dw, db, param


class rMSprop(normal):
    def __init__(self, beta:float = 0.999, e = 1e-8):
        self.beta = beta
        self.rbeta = 1. - beta
        self.e = e
        
        
    def cel(self, t, dw, db, param=None):
        # param = [Sdw, Sdb]
        if param == None:
            param = [0, 0]
        param[0] = np.add(np.multiply(self.beta, param[0]), np.multiply(self.rbeta, np.power(dw, 2)))
        param[1] = np.add(np.multiply(self.beta, param[1]), np.multiply(self.rbeta, np.power(db, 2)))
        dw = np.divide(dw, np.add(np.sqrt(param[0]), self.e))
        db = np.divide(db, np.add(np.sqrt(param[1]), self.e))
        return dw, db, param


class adam(normal):
    def __init__(self, vBeta:float = 0.9, sBeta:float = 0.999, e = 1e-8):
        self.vBeta = vBeta
        self.sBeta = sBeta
        self.rVBeta = 1. - vBeta
        self.rSBeta = 1. - sBeta
        self.t = 0
        self.vBeta_t = 1. # vBeta ** t
        self.sBeta_t = 1. # sBeta ** t
        self.e = e
        
        
    def cel(self, t, dw, db, param=None):
        # param = [Vdw, Sdw, Vdb, Sdb]
        if param == None: # 第一次迭代，初始化
            param = [0, 0, 0, 0]
        if t != self.t: # 一次新的迭代
            self.vBeta_t *= self.vBeta
            self.sBeta_t *= self.sBeta
        self.t = t
        k1 = 1. / (1. - self.vBeta_t)
        k2 = 1. / (1. - self.sBeta_t)
        param[0] = np.add(np.multiply(self.vBeta, param[0]), np.multiply(self.rVBeta, dw))
        param[1] = np.add(np.multiply(self.sBeta, param[1]), np.multiply(self.rSBeta, np.power(dw, 2)))
        param[2] = np.add(np.multiply(self.vBeta, param[2]), np.multiply(self.rVBeta, db))
        param[3] = np.add(np.multiply(self.sBeta, param[3]), np.multiply(self.rSBeta, np.power(db, 2)))
        dw = np.divide(np.multiply(param[0], k1), np.add(np.sqrt(np.multiply(param[1], k2)), self.e))
        db = np.divide(np.multiply(param[2], k1), np.add(np.sqrt(np.multiply(param[3], k2)), self.e))
        return dw, db, param
        