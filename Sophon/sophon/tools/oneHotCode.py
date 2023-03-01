# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:15:55 2023

@author: Juruoer
"""
import numpy as np

def mlticlass2OneHot(target, classSize)->np.ndarray:
    m = target.shape[0]
    if target.shape != (m, 1):
        raise ValueError(f"目标数组：{target} 不符合条件")
        
    n = classSize
    rst = np.zeros(shape=(m, n), dtype=int)
    for i in range(m):
        if target[i][0] < n:
            rst[i][target[i][0]] = 1
        else:
            raise ValueError(f"请确保目标数组：{target} 的元素值为在 0 ~ {n - 1} 之间的整数")
    return rst