import gym
import pybullet_envs
from hebbian_weights_update import *
from policies import MLP_heb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from rollout import fitness_function
import numpy as np
from main import num_rules, num_params


N_EVALS = 10
SHOW_CURVES = True


results = pickle.load(open('','rb'))


if SHOW_CURVES:
    pop_mean = np.trim_zeros(results[3])
    pop_best = np.trim_zeros(results[4])
    
    plt.plot(pop_mean, label='pop_mean')
    plt.plot(pop_best, label='pop_best')
    plt.legend()
    plt.title('Training Curve')
    plt.xlabel('generations')
    plt.ylabel('score')
    plt.show()




coeffs = results[0].mu.reshape(num_rules, num_params)
inds = results[1]



eval_episodes = []
for i in range(N_EVALS):
    score = fitness_function(coeffs, inds)
    print(i, 'score:', score)
    eval_episodes.append(score)


print('Average score:', np.mean(eval_episodes))
pickle.dump(eval_episodes, open('evaluation_scores.pickle','wb'))


