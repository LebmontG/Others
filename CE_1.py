#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:51:02 2024

@author: jaydenwang
"""

import frame
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import beta

rho = np.sqrt(0.5)
n = 10000
N = int(0.2*n)
m = 1000
a,b = 2,5
l = 96
q = 0.5
x_vector = frame.x_vector


# likelihood ratio
def ell(theta_Z, Z, theta_eta_vector, eta_vector): 
    #theta_eta_vector and eta_vector are vectors with m parameters
    return np.exp(-theta_Z*Z+0.5*theta_Z**2)*np.exp(-np.dot(theta_eta_vector,eta_vector)\
                                                    +0.5*np.dot(theta_eta_vector,theta_eta_vector))
    # return np.exp(-theta_Z*Z)*np.exp(-np.dot(theta_eta_vector,eta_vector))

def new_theta_Z(N, L, l, ell, Z): # L ell Z are vectors
    # L = np.array(L)
    return np.sum((L>l).astype(int)*ell*Z)/np.sum((L>l).astype(int)*ell)
    # return np.sum(L*ell*Z)/np.sum(L*ell)

def new_theta_eta(N, L, l, ell, eta, m): # L ell are vectors; eta is values
    # L = np.array(L)
    return np.array([np.sum((L>l).astype(int)*ell*eta[:,j])/(m*np.sum((L>l).astype(int)*ell)) for j in range(m)])
    # return np.array([np.sum(L*ell*eta[:,j])/(m*np.sum(L*ell)) for j in range(m)])

def generate_pilot_samples(N, theta_Z, theta_eta_vector, x_vector, l, m, alpha):
    
    # Generate pilot samples
    Z_vector = np.random.normal(theta_Z,1,N)   
    eta_values = np.random.normal(theta_eta_vector,1,(N,m))

    # Compute the portfolio loss, L_k
    X_values = np.array([frame.X(Z_vector[k], eta_values[k]) for k in range(N)])
    LGD_values = np.array([frame.LGD(m, Z_vector[k], a, b) for k in range(N)])
    L_vector = np.array([frame.L(X_values[k], x_vector, LGD_values[k]) for k in range(N)])
    
    # (1-q)-quantile of the losses as loss level
    l_q= np.quantile(L_vector, q = 1-q) # The {(1-q)*100}% quantile of the losses L_vector

    # Compute likelihood ratio
    ell_vector = np.array([ell(theta_Z, Z_vector[k], theta_eta_vector, eta_values[k]) for k in range(N)])
    
    # Update parameters
    theta_Z = new_theta_Z(N, L_vector, l, ell_vector, Z_vector)
    theta_eta = new_theta_eta(N, L_vector, l, ell_vector, eta_values, m)
    
    return l_q, theta_Z, theta_eta, ell_vector

def Iterative_CE(n, N, x_vector, l, m, alpha):
    
    # Initialization
    theta_Z = 0
    theta_eta_vector = np.zeros(m)
    l_q = 0

    while l_q < l: # mu_L first
        l_q, theta_Z, theta_eta_vector, ell_vector = generate_pilot_samples(N, theta_Z, theta_eta_vector, x_vector, l, m, alpha)

    # Generate n sets according to final parameters
    Z_vector = np.random.normal(theta_Z,1,n)   # data type 
    eta_values = np.random.normal(theta_eta_vector,1,(n,m))
    
    
    # Compute the portfolio loss of each simulation path, and the corresponding likelihood ratio
    X_values = np.array([frame.X(Z_vector[k], eta_values[k]) for k in range(n)])
    LGD_values = np.array([frame.LGD(m, Z_vector[k], a, b) for k in range(n)])
    L_vector = np.array([frame.L(X_values[k], x_vector, LGD_values[k]) for k in range(n)]) # portfolio loss based on pilot samples
    
    ell_vector = np.array([ell(theta_Z, Z_vector[k], theta_eta_vector, eta_values[k]) for k in range(n)])
    # ell_vector = [np.exp(-theta_Z*Z_vector[k]+0.5*theta_Z**2)*np.exp(-np.dot(theta_eta_vector,eta_values[k])\
    #                                                  +0.5*np.dot(theta_eta_vector,theta_eta_vector)) for k in range(n)]
    # L_vector = L_vector * ell_vector
    # ell_vector = np.ones(n)
    
    VaR = frame.VaR_alpha(alpha, L_vector, ell_vector)
    # VaR = frame.VaR_alpha(alpha, L_vector, n)
    ES = frame.ES_alpha(alpha, L_vector, ell_vector, VaR)

    return VaR, ES
    # return L_vector

# VaR, ES = Iterative_CE(n, N, x_vector, l, m, 0.95)
# print(VaR, ES)

# LL, ellell = Iterative_CE(N, x_vector, l, m, 0.95)

# a,b = 5,2
# Z_vector = np.random.normal(0,1,n) 
# eta_values = np.random.normal(0,1,(n,m))
# X_values = np.array([X(Z_vector[k], eta_values[k]) for k in range(n)])

# LGD_values = np.array([LGD(m, Z_vector[k], a, b) for k in range(n)])
# L_vector = np.array([L(X_values[k], x_vector, LGD_values[k]) for k in range(n)]) # portfolio loss based on pilot samples
 

# EG = Iterative_CE(N, x_vector, l, m, 0.95)
# # 绘制频率直方图
# plt.hist(EG, bins=50, density=True, color='blue', edgecolor='black')

# # 添加标题和标签
# plt.title('Frequency Histogram')
# plt.xlabel('Portfolio Loss')
# plt.ylabel('Frequency')

# # 显示图形
# plt.show()


def Risk_scenarios(n,num,alpha):
    VaR_vector, ES_vector = [],[]
    for i in tqdm(range(num)):
        VaR, ES = Iterative_CE(n, N, x_vector, l, m, alpha)
        VaR_vector.append(VaR)
        ES_vector.append(ES)
    VaR_vector = np.array(VaR_vector, dtype = np.float64)
    VaR_vector = VaR_vector[VaR_vector != 0]
    ES_vector = np.array(ES_vector, dtype = np.float64)
    ES_vector = ES_vector[ES_vector != 0]
    VaR = np.mean(VaR_vector)
    ES = np.mean(ES_vector)
    # VaR_SE = np.sqrt((1 / (n * (n - 1))) * (np.sum(VaR_vector**2) - n * VaR**2))
    # ES_SE = np.sqrt((1 / (n * (n - 1))) * (np.sum(ES_vector**2) - n * ES**2))
    VaR_SE = np.std(VaR_vector)/np.sqrt(len(VaR_vector))
    ES_SE = np.std(ES_vector)/np.sqrt(len(ES_vector))
    return VaR, VaR_SE, ES, ES_SE

# ll = CE_scenarios(10000, 0.95)

alpha_values = [0.95,0.96,0.97,0.98,0.99]
# alpha_values = [0.95]

# 10000
Risk_measure = [Risk_scenarios(n, 1000, i) for i in alpha_values]

print(Risk_measure)







