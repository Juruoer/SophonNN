# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 20:06:01 2023

@author: Juruoer
"""
import numpy as np
def clamp(target, _min, _max):
    if target > _max:
        return _max
    if target < _min:
        return _min
    return target


def normalize(target, mu, sigma):
    return np.divide(np.subtract(target, mu), sigma)

