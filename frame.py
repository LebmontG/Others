#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:40:30 2024

@author: jaydenwang
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import beta

'''
variable_vector is 1-dimensional array;
variable_values is n-dimensional array, namely matrix (a list contains several arrays).
'''

m = 1000 # obligors 1000
# N = 40 # pilot samples 40000
# n = 200 # simulation paths 200000
a = 2
b = 5
q = 0.5 # quantile
l = 95


# Z = np.random.normal(0,1,1) # common factor Z
# # Z = 0.7 # fix Z first to avoid distrubing result for each run

# eta_vector = np.random.normal(0,1,m) # idiosyncratic factors

rho = 1/np.sqrt(0.5)

# default probability p_k; m is obligors
def p(m): # only depends on the number of obligors, m
    return 0.01 * (1 + np.sin(16 * np.pi * np.arange(1, m+1) / m))

# conditional default probability p_Z 
def p_Z(m,Z):
    return norm.cdf(rho * Z - (-norm.ppf(p(m))) / rho) # rho=np.sqrt(1-rho**2)=1/np.sqrt(0.5)

# LGD
def LGD(m,Z,a,b): #alpha and beta are a and b
    return beta.ppf(p_Z(m,Z), a, b)
    # return np.array([1]*200 + [4]*200 + [9]*200 + [16]*200 + [25]*200)

# latent default variable / credit index
def X(Z, eta_vector):
    return rho*Z+rho*eta_vector

# threshold of credit index, namely xk
x_vector = -norm.ppf(p(m))     # threshold; vector; only depends on default probability; constant for each run

# L, portfolio loss
def L(X, x, LGD):   # input vectors
    return np.sum((X > x).astype(int)*LGD)

def VaR_alpha(alpha, L_vector, ell_vector):
    
    sample = []
    for i in L_vector:
        if np.mean(ell_vector*(L_vector<=i).astype(int)) >= alpha:
            sample.append(i)

    if len(sample) == 0:
        VaR = 0
    else:
        VaR = np.min(sample)
    
    return VaR
    # return sample
    
def ES_alpha(alpha, L_vector, ell_vector, VaR):
    
    if VaR == 0:
        ES = 0
    else:
        ES = VaR + np.mean(ell_vector*L_vector*(L_vector>VaR).astype(int))/(1-alpha)
    return ES
    
        
    # sorted_losses = np.sort(L_vector)
    
    # # 计算VaR的位置
    # var_position = int(alpha * len(sorted_losses))
    
    # # 获取VaR值
    # VaR = sorted_losses[var_position]
    # return VaR


# n = 10000
# m = 1000
# Z = np.zeros(n)
# Z = np.random.normal(0,1)

# default_probability = p(m)
# conditional_default_probability = p_Z(m, Z)
# loss_given_default = LGD(m, Z, a, b)
# default_variable = X(Z, np.random.normal(loc=0,size=m))
# L_vector= np.array(L(default_variable,x_vector,loss_given_default))
# ell_vector=np.ones(n)

# desired loss level, and expected portfolio loss
# mu_L = []
# for _ in tqdm(range(n)):
#     Z = np.random.normal(0,1,1)
#     eta_vector = np.random.normal(0,1,m)
#     X_vector = X(Z, eta_vector) 
#     LGD_vector = LGD(m,Z,a,b) 
#     mu_L.append(L(X_vector,x_vector,LGD_vector))
# mu_L = np.mean(mu_L) 
# l = 3*mu_L 

# L_vector = [1,3,2,0.5]
# ell_vector = [0.5,1,0.4,0.2]
# combined_list = sorted(zip(L_vector, ell_vector), key=lambda x: x[0])

# def VaR_alpha(alpha, L_vector, ell_vector):
    
#     sample = []
#     for i in L_vector:
#         value = np.mean(ell_vector*(L_vector<=i).astype(int))
#         if value >= alpha:
#             sample.append(i)
        
#     return np.min(sample)

    # # Method 1
    # # Merge L_vector and ell_vector into one list and sort them ascendingly
    # combined_list = sorted(zip(L_vector, ell_vector), key=lambda x: x[0])
    # cumulative_ell_sum = 0
    # # find min x
    # for loss, ell in combined_list:
    #     cumulative_ell_sum += ell / n
    #     # cumulative_ell_sum += ell*(loss<=l).astype(int) / n
    #     if cumulative_ell_sum >= alpha:
    #         return loss  
    # return loss  

    # # Method 2
    # sorted_losses = np.sort(L_vector)
    
    # # 计算VaR的位置
    # var_position = int(alpha * len(sorted_losses))
    
    # # 获取VaR值
    # VaR = sorted_losses[var_position]
    # return VaR

# def VaR_alpha(alpha, L_vector, n):
#     # 将损失从小到大排序
#     sorted_losses = np.sort(L_vector)
    
#     # 计算VaR的位置
#     var_position = int(alpha * len(sorted_losses))
    
#     # 获取VaR值
#     VaR = sorted_losses[var_position]
#     return VaR

# VaR_alpha(0.95, L_vector, ell_vector, 4)


# def ES_alpha(alpha, L_vector, ell_vector, n, VaR):
#     # exceed_loss = np.array([(ell * (loss - VaR)) for loss, ell in zip(L_vector, ell_vector) if loss > VaR])
#     # es_alpha = VaR + sum(exceed_loss) / (n * (1 - alpha))
#     # return es_alpha
#     # return np.mean(np.maximum(L_vector-VaR,0))
#     L_vector = L_vector*(L_vector>VaR).astype(int)
#     L_vector = L_vector[L_vector>0]
#     return np.mean(L_vector)

# ES_alpha(0.95, L_vector, ell_vector, 4, 3)










































