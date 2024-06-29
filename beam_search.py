import json

from tqdm import trange

from utils import evaluate_solution
import numpy.random as nprand
from numpy import mean
import heapq

n_cars = 100
n_roads = 2
timesteps = 30
beta_dist_alpha = 5
beta_dist_beta = 5
nprand.seed(0)
car_dist_norm = nprand.beta(
    beta_dist_alpha,
    beta_dist_beta,
    size=n_cars,
)
car_dist_arrival = list(
    map(
        lambda z: round(
            (z - min(car_dist_norm))
            / (max(car_dist_norm) - min(car_dist_norm))
            * timesteps
        ),
        car_dist_norm,
    )
)

with open("solution_pool.json", "r") as f:
    solution_pool = json.load(f)


def expand_solution(sol):
    expanded_solutions = []
    for n, item in enumerate(sol):
        if n == 0:
            continue
        new_sol = sol[:n] + [2.0 if item == 1.0 else 1.0] + sol[n + 1 :]
        expanded_solutions.append(new_sol)
    return expanded_solutions


beam = []

best_solution_fitness = 100
best_solution_out = []

for sol_pair in solution_pool:
    solution = sol_pair[0]
    fitness = -sol_pair[1]
    heapq.heappush(beam, (fitness, solution))


for x in trange(50000):
    # beam = heapq.nsmallest(10000, beam)
    beam_top = heapq.heappop(beam)
    fitness = beam_top[0]
    solution = beam_top[1]
    exp_sols = expand_solution(solution)
    for sol in exp_sols:
        # print(len(sol), len(car_dist_arrival))
        heapq.heappush(beam, ((-evaluate_solution(sol, car_dist_arrival)), sol))

    # print(best_solution_fitness)
    if fitness < best_solution_fitness:
        best_solution_fitness = fitness
        best_solution_out = solution

print(best_solution_fitness, best_solution_out)
