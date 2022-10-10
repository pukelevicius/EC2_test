

import sys
sys.path.insert(0, 'evoman')
sys.path.append('C:/Users/doman/OneDrive/files/VU AI/evolutionary computing/evoman_framework')
sys.path.append('C:/Users/doman/OneDrive/files/VU AI/evolutionary computing/evoman_framework/evoman')


from environment import Environment
from demo_controller import player_controller

# import other librairies
import random
import deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functools import partial
from operator import attrgetter
import time
import numpy as np
from math import fabs, sqrt
import glob, os
from inspect import getmembers, isfunction

import pandas as pd
from deap import tools
import matplotlib
import seaborn as sns
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt


# remove visuals
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'generalist_assignment2'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  enemies=[1,2,3],
                  multiplemode="yes",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  randomini='yes')

env.state_to_log()  # checks environment state
ini = time.time()  # sets time marker
IND_SIZE = (env.get_num_sensors() + 1) * n_hidden_neurons + (
        n_hidden_neurons + 1) * 5  # number of weights (W1, W2) for 10 hidden neurons


# define individual, fitness and strategy as datatypes
creator.create("FitnessMax", base.Fitness, weights=(
    1.0,))  # positive weight since it is not an error function, better solution = higher fitness score
creator.create("Individual", list, typecode="d",
               fitness=creator.FitnessMax)  # strategy for mutation over individual, defined later
creator.create("Strategy", list, typecode="d")

toolbox = base.Toolbox()


def generate_individual(individual, IND_SIZE):
    # Initializes individual of size n and populates it with randomly initialized strategy vector for mutation
    individual = individual(random.uniform(dom_l, dom_u) for _ in range(IND_SIZE))
    return individual


def fitness(env, individual):
    # return fitness value of one run of the game for an individual/solution with weights x
    f, p, e, t = env.play(pcont=individual)
    return (f,)  # needs to return a tuple



###### register functions ######

# generation of an individual
toolbox.register("individual", generate_individual, creator.Individual,
                 265)  # registers function 'individual', change to IND_SIZE
# generation of a population, e.g. set of individuals : returns as a list
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)  # registers function 'population', using initRepeat, returns list
# evaluation function
toolbox.register("evaluate", fitness, env)

###### register operators #######

# Mating strategy (of the parents), e.g. blending crossover
toolbox.register("mate", tools.cxUniform, indpb=0.05)

# Mutation strategy
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)

# Selection strategy : Tournament selection

toolbox.register("select", tools.selTournament, tournsize=6)



# register statistics for an individual
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

############# OPTIMIZATION  ###############
# parameters
MU = 30  # size of parent population
LAMBDA = 45 # size of generated offsprings
cxpb = 0.6  # probability of crossover
mutpb = 0.4  # probability of mutating
ngen = 4  # number of generations
dom_u = 1
dom_l = -1
nrep = 2  # number of times the experiment is repeated
k=6


#### REPEAT 10 TIMES ####
logbooks = {}
final_populations = {}
avrgs_cxrep = np.zeros((nrep, ngen))  # average across repetition
stds_cxrep = np.zeros((nrep, ngen))
maxs_cxrep = np.zeros((nrep, ngen))
gens_cxrep = np.zeros((nrep, ngen))
enems_cxrep = np.zeros((nrep, ngen))

# random_enemy = random.sample(range(1, 8), 3) - use fixed enemies instead
enemies = ['enemy1', 'enemy2', 'enemy3']

best_of_gens = np.zeros((nrep, 265))  # keep best solution from each generation to test

# store
avrgs = []
stds = []
maxs = []
gens = []
enems = []
run = []


#for enemy in range(len(enemies)):
 #   ('-------------- TRAINING AGAINST ENEMY {}--------------'.format(enemy))
# update the enemy
env.update_parameter('enemies', [1,2,3])

for rep in range(nrep):
    print('---------- TRAINING REPETITION # {}----------'.format(rep + 1))
    hof = tools.HallOfFame(1)  # store best k individuals across each generation for testing
    # Initialize the population
    population = toolbox.population(n=MU)  # initializes population with size MU
    final_populations[rep], logbooks[rep] = algorithms.eaMuPlusLambda(population, toolbox, mu=MU, lambda_=LAMBDA,
                                                                        cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                                                        halloffame=hof, stats=stats,
                                                                        verbose=False)  # this returns the final population (list of 265 weights)
    top_ind = hof[0]
    best_of_gens[rep, :] = top_ind

    # get stats in useful format for later (dataframe)
    for gen in range(ngen):
        gens.append(int(gen + 1))
        #enems.append(enemy+1)
        run.append(rep + 1)
        avrgs.append(logbooks[rep][gen]['avg'])  # mean values for one repetition
        maxs.append(logbooks[rep][gen]['max'])
        stds.append(logbooks[rep][gen]['std'])

import pandas as pd
data = pd.DataFrame( {'average fitness': avrgs, 'max fitness': maxs, 'generations': gens
                     })

data.to_csv('test_run.csv')
