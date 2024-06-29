import json
from collections import Counter, deque
from functools import partial

import tqdm
import numpy.random as nprand
from numpy import mean, quantile, median, max
from numpy import min as nmin

from utils import volume_delay_function, evaluate_solution, reduced_evaluate_solution

import pygad

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

print(sorted(car_dist_arrival))


# print(evaluate_solution([2] + [1 for _ in range(len(car_dist_arrival)-1)], car_dist_arrival))
def fitness_func(ga_instance, solution, solution_idx):
    return reduced_evaluate_solution(solution, car_dist_arrival, seq_decisions=True)


solution_pool = []

gene_space = [1, 2]
fitness_function = fitness_func

num_generations = 100

sol_per_pop = 200
num_parents_mating = round(sol_per_pop / 4)
keep_elitism = num_parents_mating
num_genes = 100

init_range_low = -2
init_range_high = 5

parent_selection_type = "tournament"
keep_parents = round(sol_per_pop / 10)

# Change this back to uniform if you want to keep it that way
# I changed it to scattered because it makes sense to increase diversity in our populations
crossover_type = "scattered"

mutation_type = "random"
mutation_percent_genes = 10


def on_generation_progress(ga):
    pbar.update(1)


for x in range(10):
    with tqdm.tqdm(total=num_generations) as pbar:
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
            gene_space=gene_space,
            on_generation=on_generation_progress,
            keep_elitism=keep_elitism,
            suppress_warnings=True,
        )

        ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=-solution_fitness
        )
    )
    # ga_instance.plot_fitness()
    solution_pool.append((list(solution), solution_fitness))
    # evaluate_solution(solution, car_dist_arrival, post_eval=True)

with open("solution_pool.json", "w+") as f:
    json.dump(solution_pool, f)
