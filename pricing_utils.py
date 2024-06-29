import random
from collections import Counter, deque, defaultdict
from functools import partial
from random import choices

import numpy as np
from numpy import mean, quantile, median
from numpy import min as nmin
from numpy import max as nmax

timesteps = 30

# random.seed(0)

def quantal_decision(routes):
    # We pass in a list of 2-tuple - (road, utility) for each road.
    utility = [u[1] for u in routes]
    utility = [u - max(utility) for u in utility]
    quantal_weights = shortform_quantal_function(utility)
    choice = choices(routes, weights=quantal_weights)
    # print("101:", [q/sum(quantal_weights) for q in quantal_weights], utility)
    return choice[0]

def shortform_quantal_function(utilities, lambd=0.5):
    return [1/(1 + np.exp(lambd * ((2*u) - sum(utilities)))) for u in utilities]

def get_utility(travel_time, econom_cost, vot):
    return -((vot * travel_time) + econom_cost)

def generate_utility_funct(travel_time, econom_cost):
    return partial(get_utility, travel_time, econom_cost)