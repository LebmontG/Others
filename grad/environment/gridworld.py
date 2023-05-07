"""
GridWorld Environment
Z.Gan 2022.10.23
"""

import gym
import time
#from . import rendering
import rendering
import numpy as np
import matplotlib.pyplot as plt

class Grid(object):
    def __init__(self, x:int = None,y:int = None,
                 reward=0.0,value=0.0,color="blue",category=0):
        self.x = x
        self.y = y
        self.color=color
        self.category=category
        self.reward = reward    # instant reward for an agent entering this grid cell
        self.value = value    # the value of this grid cell
        self.name = None    # name of this grid
        return

class GridWorld(gym.Env):
    def __init__(self,height:int=7,width:int = 7,
                 reward = 0,noise=False):
        self.width = width    # width of the env calculated by number of cells.
        self.height = height    # height
        self.reward = reward
        self.grids =[]
        self.noise = noise
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(self.height * self.width)
        self.ends = [(self.height-1,self.width-1)]
        self.start = (0,0)
        self.viewer = None
        self.state=None
        self.num_obstacle=0
        #{0:up,1:bottom,2:left,3:right}
        self.action=None
        np.random.seed(int(time.time()))
        self.reset()
        return
    def reset(self):
        self.state=0
        self.action=None
        self.grids =[]
        for i in range(self.height):
            self.grids.append([Grid(i,j,self.reward) for j in range(self.width)])
        return self.state
    def shuffle(self):
        pass
        return
    def transform(self, s):
        x = s % self.width
        y = s// self.width
        return x,y
    def is_end(self, x, y):
        if y is not None:xx, yy = x, y
        elif isinstance(x, int):xx, yy =self.state%self.width,self.state//self.width
        else:
            assert(isinstance(x, tuple)),"incomplete coordinate values"
            xx ,yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:return True
        return False
    def get_reward(self,ox,oy,y,x):
        if (y,x) in self.ends:reward=10
        else:
            noise=np.random.normal(loc=0.0, scale=0.1, size=1)[0]
            reward=-1+noise
        return reward
    def step(self, action):
        assert self.action_space.contains(action),action
        if np.random.rand()<0.05:self.action=np.random.randint(self.action_space.n)
        else:self.action = action
        old_x, old_y = self.transform(self.state)
        new_x, new_y = old_x, old_y
        if action == 2: new_x -= 1    # left
        elif action == 3: new_x += 1    # right
        elif action == 0: new_y += 1    # up
        elif action == 1: new_y -= 1    # down
        # boundary
        if new_x < 0: new_x = 0
        if new_x >= self.width: new_x = self.width-1
        if new_y < 0: new_y = 0
        if new_y >= self.height: new_y = self.height-1
        # wall effect, obstacles or boundary
        self.reward = self.get_reward(old_x, old_y,new_y,new_x)
        if self.grids[new_y][new_x].category==1:new_x, new_y = old_x, old_y
        done = self.is_end(new_x, new_y)
        self.state = self.width*new_y+new_x
        info = {"x":new_x,"y":new_y, "grids":self.grids[new_y][new_x]}
        return self.state, self.reward, done, info
    def render(self,close=False,size=30,mode="human"):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        u_size = size
        m = 2 #interval
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.width, self.height)
            for x in range(self.width):
                for y in range(self.height):
                    v = [(x*u_size+m, y*u_size+m),
                         ((x+1)*u_size-m, y*u_size+m),
                         ((x+1)*u_size-m, (y+1)*u_size-m),
                         (x*u_size+m, (y+1)*u_size-m)]
                    rect = rendering.FilledPolygon(v)
                    r = self.grids[y][x].reward/10
                    if r < 0:
                        rect.set_color(0.9-r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9,0.9,0.9)
                    self.viewer.add_geom(rect)
                    # draw frameworks
                    v_outline = [(x*u_size+m, y*u_size+m),
                                     ((x+1)*u_size-m, y*u_size+m),
                                     ((x+1)*u_size-m, (y+1)*u_size-m),
                                     (x*u_size+m, (y+1)*u_size-m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)
                    if self.is_end(x,y):
                        # give end state cell a yellow outline.
                        outline.set_color(0.9,0.9,0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids[y][x].category == 1:
                        #obstacle cells are with gray color
                        rect.set_color(0.3,0.3,0.3)
                    else:pass
            # draw agent
            self.agent = rendering.make_circle(u_size/4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)
        # update position of an agent
        x, y = self.transform(self.state)
        self.agent_trans.set_translation((x+0.5)*u_size, (y+0.5)*u_size)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if __name__ =="__main__":
    env = GridWorld()
    env.reset()
    nfs = env.observation_space
    nfa = env.action_space
    print("nfs:%s; nfa:%s"%(nfs,nfa))
    print(env.observation_space)
    print(env.action_space)
    print(env.state)
    #env.render()
    #x = input("press any key to exit")
    for _ in range(20):
        #env.render()
        a = env.action_space.sample()
        state, reward, isdone, info = env.step(a)
        print("{0}, {1}, {2}, {3}".format(a, reward, isdone, info))
        #print(state)
    env.close()
    print("env closed")