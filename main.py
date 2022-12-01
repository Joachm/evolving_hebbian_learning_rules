from policies import *
from rollout import *
import gym
import pybullet
import pybullet_envs
from ES_classes import *
import concurrent.futures
import multiprocessing
import pickle
gym.logger.set_level(40)
torch.set_num_threads(1)

EPOCHS = 1600 
popsize = 512
cpus = 64
if cpus==-1:
    cpus = multiprocessing.cpu_count()

start_sig = 0.1
plastic_weights =(128*28)+(64*128)+(8*64) # depending on network size
num_rules = plastic_weights  #can be any number <= plastic_weights
num_params = 5 #number of parameters in the hebbian rule
save_every = 500


if num_rules == plastic_weights:
    inds = np.arange(plastic_weights)
else:
    inds = np.random.randint(0,num_rules, (plastic_weights))


coeffs = np.random.normal(0,start_sig, (num_rules, num_params))

##initialize solver
#solver = CMAES(coeffs.reshape(-1), popsize=popsize)

#'''
solver = OpenES(len(coeffs.reshape(-1)),
        popsize=popsize,
        rank_fitness=True,
        learning_rate=0.01,
        learning_rate_decay=0.9999,
        learning_rate_limit=0.001,
        sigma_init=0.1,
        sigma_decay=0.999,
        sigma_limit=0.01)

solver.set_mu(coeffs.reshape(-1))
#'''


def worker(arg):
    fit_func, coeffs, inds = arg
    r  = fit_func(coeffs, inds)
    return r

pop_mean_curve = np.zeros((EPOCHS))
best_sol_curve = np.zeros((EPOCHS))
evals = []
print('Begin Training')
for epoch in range(EPOCHS):
    
    solutions = solver.ask()
    coeffs  = [solution.reshape(num_rules,num_params) for solution in solutions]
        
    with concurrent.futures.ProcessPoolExecutor(cpus) as executor: 
        work_args = [(fitness_function, coeff, inds) for coeff in coeffs]
        fitlist = executor.map(worker, work_args)
        

    fitlist = list(fitlist) 
    solver.tell(fitlist)

    print('epoch:', epoch)
    pop_mean_curve[epoch] = np.mean(fitlist)
    best_sol_curve[epoch] = np.max(fitlist)
    print(pop_mean_curve[epoch])
    print(best_sol_curve[epoch])
    if (epoch+1)%save_every == 0 :

        print('saving')
        print('mean score',np.mean(fitlist))
        print('best score', np.max(fitlist))

        pickle.dump((solver,
            inds, epoch,
            pop_mean_curve,
            best_sol_curve),open('trained_'+str(num_rules)+'_'+str(num_params)+'_'+str(epoch)+'_'+str(np.mean(fitlist))+ '.pickle', 'wb'))

 
    print()
