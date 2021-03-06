'''
Adapted from E. Najarro
https://github.com/enajx/HebbianMetaLearning
'''
import gym
import pybullet
from hebbian_weights_update import *
from policies import MLP_heb
import torch
import torch.nn as nn
import matplotlib.pyplot


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight.data, -0.1, 0.1)

def fitness_function(coeffs, inds):
    env = gym.make('AntBulletEnv-v0')
    with torch.no_grad():

        inp_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        NNet = MLP_heb(inp_size, action_size)
        NNet.apply(weights_init)
        
        w1, w2, w3 = list(NNet.parameters())

        w1 = w1.detach().numpy()
        w2 = w2.detach().numpy()
        w3 = w3.detach().numpy()

        obs = env.reset()

        done = False
        total_reward = 0
        while True:
            

            o0,o1,o2, action  = NNet(obs)
            obs, reward, done, info = env.step(action)

            o0 = o0.numpy()
            o1 = o1.numpy()
            o2 = o2.numpy()
            action = action.numpy()
            
            total_reward += reward
           
            if done:
                break
          
            w1, w2, w3 = hebbian_update(coeffs, w1, w2, w3, o0, o1, o2, action, inds)
    env.close()

    return total_reward
