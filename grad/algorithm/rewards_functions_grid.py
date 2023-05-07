import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def rand_color():
    return (np.random.random(), np.random.random(), np.random.random())

def str_to_list(s):
    s=s.strip("\n").strip("[]").split(",")
    return [int(c) for c in s]

def set_fig_size(plt):
    # plt.figure(figsize=(16, 18))
    plt.figure(dpi=150)
    # plt.figure().set_size_inches(6,8)
    return plt

def aver_plot(names,path):
    color_used=set()
    plt.figure(dpi=300)
    for n in names:
        p=path+n+".csv"
        if not os.path.isfile(p):continue
        f=open(p,"r")
        data=np.array([str_to_list(l) for l in f])
        f.close()
        l,e=data.shape;std=[];avg=[]
        for i in range(e):
            std.append(np.std(data[:,i]))
            avg.append(np.mean(data[:,i]))
        x=[i+1 for i in range(e)]
        #random color
        col = []
        while len(col)<2:
            t=rand_color()
            if t not in color_used:
                col.append(t)
                color_used.add(t)
        #average curve
        plt.plot(x,avg,
                 marker=".",color=col[0],
                 # markeredgecolor=col[1],
                 markeredgecolor=col[0],
                 linewidth=1,label=n)
        y1=[];y2=[]
        for i in range(e):
            y1.append(avg[i]+std[i]/2)
            y2.append(avg[i] - std[i] / 2)
        plt.fill_between(x,y1,y2,facecolor=col[0],alpha=0.2)
    #plt.xticks(x,x)
    plt.grid()
    # plt.title("")
    #plt.gcf().savefig('temp.png')
    plt.legend(loc="best")
    plt.xlabel("episode")
    plt.ylabel("steps")
    return plt
#aver_plot(rf_names,reward_path)
def clear_file(path,names):
    for n in names:
        p=path+n+".csv"
        if not os.path.isfile(p):continue
        f=open(p,"r+")
        f.truncate(0)
    return

class GT(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.style="GT"
        return

class Intrinsic(object):
    def __init__(self,state_space):
        self.d=dict()
        self.style="intrinsic"
        return
    def step(self,s):
        if s in self.d:self.d[s]+=1
        else:self.d[s]=1
        return 1/self.d[s]

#Phi
class RS1(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="rewardshaping"
        return
    def transform(self,_s,s,r):
        ox,oy = _s % self.l,_s// self.l
        x,y = s % self.l,s// self.l
        e=2*self.l
        return r+0.9*(10-(e-x-y))-(10-(e-ox-oy))
#prior knowledge
class RS2(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="rewardshaping"
        return
    def transform(self,_s,s,r):
        ox,oy = _s % self.l,_s// self.l
        x,y = s % self.l,s// self.l
        return r+0.9*(y+x)-oy-ox
#inverse
class RS3(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="rewardshaping"
        return
    def transform(self,_s,s,r):
        ox,oy = _s % self.l,_s// self.l
        x,y = s % self.l,s// self.l
        return r-0.9*(y+x)+(oy+ox)

#random
class Heuristic1(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="heuristic"
        return
    def transform(self,_s,s,r):
        return np.random.normal(loc=0.0, scale=1, size=1)
#radius to the end
class Heuristic2(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="heuristic"
        return
    def transform(self,_s,s,r):
        x = s % self.l
        y = s// self.l
        return r-((self.l-y)**2+(self.l-x)**2)/10
#inverse
class Heuristic3(object):
    def __init__(self,state_space):
        self.l=int(np.sqrt(state_space))-1
        self.end=[self.l-1,self.l-1]
        self.style="heuristic"
        return
    def transform(self,_s,s,r):
        x = s % self.l
        y = s// self.l
        return r+(x+y)

class AWSL(object):
    def __init__(self,state_space):
        self.style="AWSL"
        self.s_space=state_space
        self.weight=None
        self.glb_v=0
        self.visit_count=None
        self.s_minmax=None
        self.glb_minmax=None
        self.acc_r=0
        self.acc_sr=None
        self.alpha=0.9
        self.reset()
        return
    def reset(self):
        self.glb_v=0
        self.acc_r=1
        self.acc_sr=np.ones(self.s_space)
        self.weight=np.ones(self.s_space)
        self.visit_count=np.zeros(self.s_space)
        self.s_minmax=np.zeros((self.s_space,2))
        self.glb_minmax=np.array([0,1])
        return
    def step(self,s,r):
        self.glb_v+=1
        self.visit_count[s]+=1
        self.s_minmax[s][0]=min(self.s_minmax[s][0],r)
        self.s_minmax[s][1]=max(self.s_minmax[s][1],r)
        self.glb_minmax[0]=min(self.glb_minmax[0],r)
        self.glb_minmax[1]=max(self.glb_minmax[1],r)
        t=self.s_minmax[s][1]-self.s_minmax[s][0]
        if t!=0:r_norm=(r-self.s_minmax[s][0])/t
        else:r_norm=r
        self.acc_r+=r_norm+1
        self.acc_sr[s]+=r_norm+1
        ucb=np.sqrt((np.log(self.glb_v)+1)/self.visit_count[s])
        awsl_r=(self.acc_sr[s]/self.visit_count[s]*self.glb_v/self.acc_r)
        # self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*awsl_r
        # self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*ucb
        self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*ucb*awsl_r
        # self.weight[s]=awsl_r
        return
    def W(self,s):
        return self.weight[s]
    def test(self,s):
        print("_________")
        print(s)
        print(self.glb_v,self.visit_count[s])
        print(self.s_minmax[s],self.glb_minmax)
        print(self.acc_r,self.acc_sr[s])
        print(self.weight[s])
        print("_________")

if __name__ =="__main__":
    rf=AWSL(11)
    for i in range(1,10):
        t,r=np.random.randint(i),1
        rf.step(t,r)
        rf.test(t)