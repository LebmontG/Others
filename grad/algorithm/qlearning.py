import numpy as np

class QLearning():
    def __init__(self,dim_state,dim_action):
        self.d_s,self.d_a=dim_state,dim_action
        self.qtable=np.random.normal(loc=0, scale=0.01, size=(dim_state,dim_action))
        self.gamma=0.9
        return
    def reset(self):
        self.qtable=np.random.normal(loc=0, scale=0.01, size=(self.d_s,self.d_a))
        return
    def select(self,state):
        return np.argmax(self.qtable[state])
    def optimize(self,_state,action,state,reward):
        self.qtable[_state,action]=reward+self.gamma*self.qtable[state].max()
        return

if __name__ =="__main__":
    scl=6
    agent=QLearning(scl,scl)
    r = np.array([[-1, -1, -1, -1, 0, -1],
                  [-1, -1, -1, 0, -1, 10],
                  [-1, -1, -1, 0, -1, -1],
                  [-1, 0, 0, -1, 0, -1],
                  [0, -1, -1, 0, -1, 10],
                  [-1, 0, -1, -1, 0, 10]])
    r = np.matrix(r)
    import random
    for i in range(1000):
        state = random.randint(0, scl-1)
        while state != scl-1:
            m=agent.qtable[state].max()
            ind=[i for i in range(scl) if (agent.qtable[state,i]-m)<1e-4]
            next_state =random.choice(ind)
            agent.optimize(state,next_state,next_state,r[state, next_state])
            state = next_state
    print(agent.qtable)