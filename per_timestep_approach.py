from collections import Counter, defaultdict
from functools import partial
from itertools import combinations, product
from statistics import geometric_mean
import numpy as np
import tqdm
import numpy.random as nprand
from numpy import mean, quantile, median, max
from numpy import min as nmin
from copy import deepcopy
from utils import (
    volume_delay_function,
    evaluate_solution,
    is_simulation_complete,
    alternative_get_new_travel_times,
)
from pprint import pp

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

"""
So the plan here is that we basically find the solution at each timestep that minimises the increase in the 
mean travel time for both routes. 

In theory - minimising this will have a longer term impact on the mean travel time

Do we care about the mean travel time of the vehicles at each timestep, or are we specifically decreasing the travel time
at the end of the timestep for each vehicle? Which will have the best outcome.

"""
solution = []
time = 0

roadQueues = {r: [] for r in [1, 2]}
roadVDFS = {
    1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
    2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
}
roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
arrival_timestep_dict = Counter(car_dist_arrival)
arrived_vehicles = []

time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}

while not is_simulation_complete(roadQueues, time):
    # update the arrived vehicles, get accurate travel times
    leavers = 0
    roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
    )

    # print("Vehicles on road:", {r: len(x) for r, x in roadQueues.items()})
    # print("Actual:", roadTravelTime)

    # print("line :92", time, roadTravelTime[road], {r: len(z) for r, z in roadQueues.items()})

    num_vehicles_arrived = arrival_timestep_dict[time]
    possible_solutions = list(product([1, 2], repeat=num_vehicles_arrived))
    best_partial_solution = []
    best_partial_cost = np.inf
    best_updated_travel_time = {}
    best_expected_leavers = []
    solution_space = {}
    for possible_solution in possible_solutions:
        selections = Counter(possible_solution)
        # TODO: here we need to change how we calculate travel time
        """
        alternative_get_new_travel_times(roadQueues, roadVDFS, time,
                                         arrived_vehicles, time_out_car)
        For each possible outcome, we should generate new values for each of these and see what the new travel time 
        Of the roads will be.                                 
        """
        potential_time = time + 1
        potential_road_queues = deepcopy(roadQueues)
        potential_arrived_vehicles = deepcopy(arrived_vehicles)
        potential_time_out_car = deepcopy(time_out_car)

        for road, count in selections.items():
            potential_road_queues[road] = potential_road_queues[road] + [
                (
                    road,
                    time,
                    time + roadTravelTime[road],
                    time + roadTravelTime[road],
                )
                for _ in range(count)
            ]
            potential_time_out_car[road][round(time + roadTravelTime[road])] = (
                    potential_time_out_car[road][round(time + roadTravelTime[road])] + count
            )

        next_ts_travel_time = alternative_get_new_travel_times(
            potential_road_queues,
            roadVDFS,
            potential_time,
            potential_arrived_vehicles,
            potential_time_out_car,
        )[0]
        solution_score = mean(list(next_ts_travel_time.values()))
        solution_space[possible_solution] = solution_score
        # print(solution_score, possible_solution)
        if solution_score < best_partial_cost:
            best_partial_solution = possible_solution
            best_partial_cost = solution_score
            best_updated_travel_time = next_ts_travel_time

    # if num_vehicles_arrived > 0:
    #     pp(dict(sorted(solution_space.items(), key=lambda x: x[1])))

    for decision in best_partial_solution:
        roadQueues[decision] = roadQueues[decision] + [
            (
                decision,
                time,
                time + roadTravelTime[decision],
                time + roadTravelTime[decision],
            )
        ]
        # print(decision)
        print(
            "I am vehicle",
            "___",
            ". I am choosing",
            decision,
            "at time",
            time,
            "with travel time",
            roadTravelTime[decision],
            "and ETA of",
            time + roadTravelTime[decision],
            ". The alternative is",
            (1 if decision == 2 else 2),
            "with travel time",
            roadTravelTime[(1 if decision == 2 else 2)],
            "and ETA of",
            time + roadTravelTime[(1 if decision == 2 else 2)],
        )
        solution.append(decision)
        time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
        )
    # print({r: len(x) for r,x in roadQueues.items()})
    time += 1
print([(c[1], c[3]) for c in arrived_vehicles])
print(time)
print(len(solution), len(car_dist_arrival), len(arrived_vehicles))
print(evaluate_solution(solution, car_dist_arrival, post_eval=True, seq_decisions=True), solution)
