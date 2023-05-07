import numpy as np

class GT_cont(object):
    def __init__(self):
        self.style="GT"
        return

class AWSL_cont(object):
    def __init__(self):
        self.style="AWSL"
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
        self.acc_sr=dict()
        self.weight=dict()
        self.visit_count=dict()
        self.s_minmax=dict()
        self.glb_minmax=np.array([0,1])
        return
    def step(self,s,r):
        s=round(s[2],1)
        self.glb_v+=1
        if s in self.visit_count:self.visit_count[s]+=1
        else:self.visit_count[s]=1
        if s in self.s_minmax:
            self.s_minmax[s][0]=min(self.s_minmax[s][0],r)
            self.s_minmax[s][1]=max(self.s_minmax[s][1],r)
        else:
            self.s_minmax[s]=[r,r]
        self.glb_minmax[0]=min(self.glb_minmax[0],r)
        self.glb_minmax[1]=max(self.glb_minmax[1],r)
        t=self.s_minmax[s][1]-self.s_minmax[s][0]
        if t!=0:r_norm=(r-self.s_minmax[s][0])/t
        else:r_norm=r
        self.acc_r+=r_norm+1
        if s in self.acc_sr:self.acc_sr[s]+=r_norm+1
        else:self.acc_sr[s]=r_norm+1
        ucb=np.sqrt((np.log(self.glb_v)+1)/self.visit_count[s])
        awsl_r=(self.acc_sr[s]/self.visit_count[s]*self.glb_v/self.acc_r)
        # self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*awsl_r
        # self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*ucb
        if s in self.weight:self.weight[s]=self.alpha*self.weight[s]+(1-self.alpha)*ucb*awsl_r
        else:self.weight[s]=ucb*awsl_r
        # self.weight[s]=awsl_r
        return
    def W(self,s):
        s=round(s[2],1)
        if s in self.weight:return self.weight[s]
        else:return 1
    def test(self,s):
        print("_________")
        print(s)
        print(self.glb_v,self.visit_count[s])
        print(self.s_minmax[s],self.glb_minmax)
        print(self.acc_r,self.acc_sr[s])
        print(self.weight[s])
        print("_________")

if __name__ =="__main__":
    rf=AWSL_cont()
    rf.step([1,2,3,4],1)
    rf.W([1,2,3,4])