'''
Prediction of static-torque in Bolt-tightening by robots.
Zh Gan 2022.1.13
'''

import sys
sys.path.append("C:\\anaconda2\\pkgs\\tensorflow-base-2.3.0-eigen_py37h17acbac_0\\Lib\\site-packages\\tensorflow")
sys.path.append("C:\\anaconda2\\pkgs\\tensorflow-base-2.3.0-eigen_py37h17acbac_0\\Lib\\site-packages")
from scipy import interpolate
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import xgboost as xgb
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

#data processing
x=[]
y=[]
xi=[]
y_dirty=[]
num=249
for name in os.listdir('train'):
    file=pd.read_csv('train\\'+name)
    v=file['value'][0]
    #plt.plot(file['x'],file['y'])
    #delete data in first phase
    file=file.drop(file[file['y']<10].index)
    #delete data 'y' descending
    #file.drop(file[file['y']<10].index)
    l=len(file['x'])
    if l in range(100,num):
        y.append(v)
        X=range(1,l+1)
        f1=np.array(pd.DataFrame(file,columns=['x'])).T
        f2=np.array(pd.DataFrame(file,columns=['y'])).T
        f=interpolate.interp1d(X,f1,kind="cubic")
        f1=f(np.linspace(1,l,num-1))
        f=interpolate.interp1d(X,f2,kind="cubic")
        f2=f(np.linspace(1,l,num-1))
        x.append(np.concatenate((f1,f2)))
    elif l<100:
        y_dirty.append(v)
x,y=np.array(x),np.array(y)
index=np.random.choice(np.arange(len(x)),size=33,replace=False)
x_t=x[index]
y_t=y[index]
index=np.delete(range(len(x)),index)
x=x[index]
y=y[index]
x_torque=[x[i][1] for i in range(len(x))]
x_angle=[x[i][0] for i in range(len(x))]
x_torque_t=[x_t[i][1] for i in range(len(x_t))]
x_angle_t=[x_t[i][0] for i in range(len(x_t))]
x_torque,x_angle=np.array(x_torque),np.array(x_angle)
x_torque_t,x_angle_t=np.array(x_torque_t),np.array(x_angle_t)
xi=[x_torque[i] for i in range(len(x_torque)) if y[i]>27 or y[i]<24]
yi=[y[i] for i in range(len(y)) if y[i]>27 or y[i]<24]
#v=[(y[i],max(x[i][:,1])) for i in range(len(x))]
#v=[[y[i],max(x[i][:,0])] for i in range(len(x))]
#l=[len(x[i]) for i in range(len(x))]
#l.sort()
#num=l[-15]
#train_data=[(x[i],y[i]) for i in range(367)]
#test_data=[(x[i],y[i]) for i in range(367,400)]
#x,x_t,y,y_t=sklearn.model_selection.train_test_split(x,y,test_size= 0.1,shuffle=True)
#x,x_t=x[:-33],x[-33:]
#x_torque,x_torque_t=x_torque[:-33],x_torque[-33:]
#x_angle,x_angle_t=x_angle[:-33],x_angle[-33:]
#y,y_t=y[:-33],y[-33:]
#a=np.concatenate((x_torque,x_angle),1)

#GBDT
params = {
    "n_estimators":500,
    "max_depth":20,
    "min_samples_split":50,
    "learning_rate": 0.003
}
gbdt=GradientBoostingRegressor(**params)
gbdt.fit(x_torque,y)
er=metrics.mean_squared_error(y_t,gbdt.predict(x_torque_t))
print(er)
#gbdt.fit(x_angle,y)
#er=metrics.mean_squared_error(y_t,gbdt.predict(x_angle_t))

#xgboost
#for i in range(5):
params = {
    "max_depth":7,
    "booster":'gbtree',
    "n_estimators":6000,
    "learning_rate":0.2,
    "objective":'reg:linear',
    "seed":1000
}
xgboost=xgb.XGBRegressor(**params)
xgboost.fit(x_torque,y)
er=metrics.mean_squared_error(y_t,xgboost.predict(x_torque_t))
#er=metrics.mean_squared_error(y,xgboost.predict(x_torque))
print(er)

#nerualnetwork
class NN(nn.Module):
    def __init__(self, input_dim=248,output_dim=1):
        super(NN, self).__init__()
        self.lr=0.00003
        self.epoches=20
        self.batchsize=1
        #self.co1 = nn.Conv1d(1,1,5,stride=1)
        self.fc1 = nn.Linear(input_dim,124)
        self.fc2 = nn.Linear(124,62)
        self.fc3 = nn.Linear(62,21)
        self.fc4 = nn.Linear(21,6)
        self.fc5 = nn.Linear(6,output_dim)
        self.optimizer=optim.Adam(self.parameters(),self.lr)
        self.loss=nn.MSELoss()
        self.losses=[]
        #self.optimizer.param_groups
    def train(self,x,y):
        for i in range(self.epoches):
            loc=0
            L=0
            while(loc+self.batchsize<len(y)):
                self.optimizer.zero_grad()
                l=self.loss(self.forward(x[loc:loc+self.batchsize]),y[loc:loc+self.batchsize])
                l.backward()
                self.optimizer.step()
                loc+=self.batchsize
                L+=float(l)
            self.losses.append(L)
        return
    def forward(self, x):
        # 各层对应的激活函数
        #x=self.co1(x)
        #x=self.fc7(x)
        #x=self.fc6(x)
        x =self.fc1(x)
        x =(self.fc2(x)) 
        x =self.fc3(x)
        x = (self.fc4(x))
        x=self.fc5(x)
        return x
m=NN()
m.train(torch.FloatTensor(x_torque),torch.FloatTensor(y.T))
y_h=m.forward(torch.FloatTensor(x_torque_t))
#y_h=m.forward(torch.FloatTensor(x_torque))
er=torch.FloatTensor(y_t)-y_h.T
print((er[0]*er[0].T).sum()/len(y_t))
#er=sum(abs(torch.FloatTensor(y)-y_h.T))

#output
me=np.mean(y_dirty)
for name in os.listdir('test'):
    file=pd.read_csv('test\\'+name)
    file=file.drop(file[file['y']<10].index)
    l=len(file['y'])
    if l==0:
        res=pd.DataFrame([[name[:3],me]])
        res.to_csv('submit.csv',sep=',',mode='a',header=False,index=False)
        continue
    X=range(1,l+1)
    x=np.array(pd.DataFrame(file,columns=['y'])).T
    f=interpolate.interp1d(X,x,kind="cubic")
    x=f(np.linspace(1,l,num-1))
    res=pd.DataFrame([[name[:3],m.forward(torch.FloatTensor(x)).item()]])
    #res=pd.DataFrame([[name[:3],gbdt.predict(x)[0]]])
    res.to_csv('submit.csv',sep=',',mode='a',header=False,index=False)
