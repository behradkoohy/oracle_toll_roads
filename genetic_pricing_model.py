from collections import Counter, defaultdict, deque
from functools import partial
from numpy import mean, quantile, median
from numpy import min as nmin
from numpy import max as nmax
import tqdm
import numpy.random as nprand
from pygad import pygad


from pricing_utils import generate_utility_funct, quantal_decision
from utils import (
    volume_delay_function,
    is_simulation_complete,
    alternative_get_new_travel_times,
    reduced_is_simulation_complete,
)


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

car_vot = nprand.uniform(2.5, 9.5, n_cars)

bound = 1

pricing_dict = {-1: lambda x: x - bound, 0: lambda x: x, 1: lambda x: x + bound}


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
print(len(car_dist_arrival))
solution = [1 for _ in range(60+2)]
time = 0


def evaluate_solution(solution, car_dist_arrival, post_eval=False, seq_decisions=False):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """
    # if len(solution) != len(car_dist_arrival):
    #     raise Exception("Length of solution and car_dist_arrival must be equal")
    if len(solution) != (timesteps+1)*2:
        raise Exception(
            "Length of solution needs to be the same as the number of timesteps * road"
        )

    roadQueues = {r: [] for r in [1, 2]}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    arrived_vehicles = []
    roadPrices = {r: 20.0 for r in roadVDFS.keys()}

    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

    arrived_vehicles = []
    time = 0

    sol_deque = deque(solution)
    car_vot_deque = deque(car_vot)
    while not reduced_is_simulation_complete(roadQueues, time):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        roadPrices = {
            r: pricing_dict[sol_deque.popleft()](roadPrices[r]) for r in roadVDFS.keys()
        }
        roadPrices = {
            r: 1 if x < 1 else x for r,x in roadPrices.items()
        }

        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        # if num_vehicles_arrived is None:
        #     num_vehicles_arrived = 0
        # just collect the decision of the vehicles
        # decisions = [sol_deque.popleft() for _ in roadVDFS.keys()]

        cars_arrived = [car_vot_deque.popleft() for _ in range(num_vehicles_arrived)]
        # We need to change this so cars make the decision quantally, and we adjust the pricing instead
        road_partial_funct = {
            r: generate_utility_funct(roadTravelTime[r], roadPrices[r])
            for r in roadVDFS.keys()
        }
        car_utilities = {
            n: list({r: utility_funct(n_car_vot) for r, utility_funct in road_partial_funct.items()}.items())
            for n, n_car_vot in enumerate(cars_arrived)
        }
        car_quantal_decision = {c: quantal_decision(r) for c,r in car_utilities.items()}
        decisions = zip([d[0] for d in car_quantal_decision.values()], cars_arrived)

        """
        Here you need to generate the list of utilities for each vehicle/road combo
        and then put them into a decision list so they can be allocated to the correct road.
        """
        # add vehicles to the new queue
        for (decision, vot) in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    vot,
                    time + roadTravelTime[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                    time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                )
        time += 1
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [
            car for car in roadQueues[road]
        ]
    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    welfare = [(c[3] - c[1])*c[2] for c in arrived_vehicles]
    # print([(c[3], c[1], c[3] - c[1]) for c in arrived_vehicles])
    # print([c for c in arrived_vehicles if c[0] == 2])
    # travel_time = np.asarray(travel_time, dtype=np.float64)
    if post_eval:
        print("Travel Time:",
            nmin(travel_time),
            quantile(travel_time, 0.25),
            mean(travel_time),
            median(travel_time),
            quantile(travel_time, 0.75),
            nmax(travel_time),
        )
        print("Welfare:",
              nmin(welfare),
              quantile(welfare, 0.25),
              mean(welfare),
              median(welfare),
              quantile(welfare, 0.75),
              nmax(welfare),
              )

    return -mean(travel_time)

def fitness_func(ga_instance, solution, solution_idx):
    return evaluate_solution(solution, car_dist_arrival, seq_decisions=True)

solution_pool = []

gene_space = [-1, 0, 1]
fitness_function = fitness_func

num_generations = 500

sol_per_pop = 75
num_parents_mating = round(sol_per_pop / 4)
keep_elitism = num_parents_mating
num_genes = 62

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
evaluate_solution(solution, car_dist_arrival, post_eval=True)
