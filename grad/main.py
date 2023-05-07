"""
Reward functions test in GridWorld Environment
Z.GAN 2023.2.21
"""

import sys,os,datetime
# .py files in package "environment" can import each other
sys.path.append("./environment/")
#from environment import *
import gridworld,cartpole
from reward_visualization import plot_grid_rewards
import numpy as np
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from algorithm import *

reward_path="results/"
# rf_names=["GT","AWSL",
#           "Heuristic1","Heuristic2","Heuristic3",
#           "RS1","RS2","RS3",
#           "Intrinsic"]
# rf_names=["GT_cont","AWSL_cont"]
rf_names=["GT_cont","AWSL_cont"]
h,w=8,8
average_size=1 #number of one reward function test
episode=200 #number in an entire train
exp_epi=0 #number of exploration in an entire train

def reward_function(_state,state,reward,rf):
    if rf.style=="AWSL":
        rf.step(_state,reward)
        reward*=rf.W(_state)
    elif rf.style=="intrinsic":reward+=rf.step(state)
    elif rf.style=="GT":return reward
    else:reward=rf.transform(_state,state,reward)
    return reward

def test(agent,env):
    env.reset()
    _state=env.state;isdone=0
    rewards=[];i=0
    while isdone!=1:
        a = agent.select(_state)
        state, reward, isdone, info = env.step(a)
        rewards.append(reward)
        _state=state
        if i>200:break
        else:i+=1
    env.close()
    return rewards

def train(env,agent,rf,episode,exp_epi):
    steps=[]
    agent.reset()
    for t in range(episode+exp_epi):
        _state=env.reset();isdone=0
        rewards=[];i=1
        if rf.style=="AWSL":rf.reset()
        while isdone!=1:
            if t<exp_epi:a = env.action_space.sample()
            else:a = agent.select(_state)
            state, reward, isdone, info = env.step(a)
            rewards.append(reward)
            reward=reward_function(_state,state,reward,rf)
            agent.optimize(_state,a,state,reward)
            _state=state
            if i>=200:break
            else:i+=1
        env.close()
        steps.append(i)
    return steps
#gridworld
# time=datetime.datetime.now().time().second*100
# env=gridworld.GridWorld(h,w)
# agent=QLearning(env.observation_space.n,env.action_space.n)
# for rf_name in rf_names:
#     f=open("results/"+rf_name+".csv","a+")
#     for _ in range(average_size):
#         time=int(str(time**2)[:4])
#         np.random.seed(time)
#         exec("rf="+rf_name+"(env.observation_space.n)")
#         steps=train(env, agent, rf,episode,exp_epi)
#         f.write(str(steps)+"\n")
#     f.close()
#cartpole
env=cartpole.CartPole()
#print(env.action_space.low,env.action_space.high)
agent=PPO(len(env.observation_space.low),len(env.action_space.low))
for rf_name in rf_names:
    f=open("results/"+rf_name+".csv","a+")
    exec("rf="+rf_name+"()")
    for _ in range(average_size):
        steps=agent.train(env,episode, rf)
        f.write(str(steps)+"\n")
    f.close()

# clear_file("results/",rf_names)
plt=aver_plot(rf_names,reward_path)
plt.gcf().savefig('RF.pdf',format='pdf')

# fig=plot_grid_rewards(h,w,agent.qtable,style="Q")
# plt.gcf().savefig('RF.pdf',format='pdf')

# for i in range(len(agent.qtable)):
#     agent.qtable[i]=np.ones(5)*rf.weight[i]
# fig=plot_grid_rewards(h,w,agent.qtable,style="V")
# plt.gcf().savefig('RF.pdf',format='pdf')
