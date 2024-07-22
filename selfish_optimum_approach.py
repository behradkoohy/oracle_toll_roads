from collections import Counter, defaultdict
from functools import partial
from itertools import combinations, product

import numpy as np
import tqdm
import numpy.random as nprand
from numpy import mean, quantile, median
from numpy import max as nmax
from numpy import min as nmin

from utils import (
    volume_delay_function,
    evaluate_solution,
    is_simulation_complete,
    alternative_get_new_travel_times, reduced_is_simulation_complete, reduced_evaluate_solution,
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


while not reduced_is_simulation_complete(roadQueues, time):
    # update the arrived vehicles, get accurate travel times
    num_vehicles_arrived = arrival_timestep_dict[time]
    timestep_decisions = []
    for vehicle in range(num_vehicles_arrived):
        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
        if roadTravelTime[1] < roadTravelTime[2]:
            quickest_road, travel_time = 1, roadTravelTime[1]
            slow_road, slow_tt = 2, roadTravelTime[2]
        else:
            quickest_road, travel_time = 2, roadTravelTime[2]
            slow_road, slow_tt = 1, roadTravelTime[1]

        roadQueues[quickest_road] = roadQueues[quickest_road] + [
            (quickest_road, time, time + travel_time, time + travel_time)
        ]
        # print(time, vehicle, roadTravelTime, quickest_road)
        # print(
        #     "I am vehicle",
        #     vehicle,
        #     ". I am choosing",
        #     quickest_road,
        #     "at time",
        #     time,
        #     "with travel time",
        #     travel_time,
        #     "and ETA of",
        #     time + travel_time,
        #     ". The alternative is",
        #     slow_road,
        #     "with travel time",
        #     slow_tt,
        #     "and ETA of",
        #     time + slow_tt,
        # )
        # print({r:x for r,x in roadTravelTime.items()})
        # print({r:len(x) for r,x in roadQueues.items()})
        solution.append(quickest_road)
        time_out_car[quickest_road][round(time + roadTravelTime[quickest_road])] = (
                time_out_car[quickest_road].get(round(time + roadTravelTime[quickest_road]), 0) + 1
        )

    if num_vehicles_arrived == 0:
        for road in roadQueues.keys():
            arrived_vehicles = arrived_vehicles + [
                car for car in roadQueues[road] if car[3] <= time
            ]
            roadQueues[road] = [car for car in roadQueues[road] if car[3] > time]
    # if time == 11:
    #     exit()
    time += 1
for road, queue in roadQueues.items():
    arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]

# print([(c[1], c[3]) for c in arrived_vehicles])
print(len(solution), len(car_dist_arrival), len(arrived_vehicles))
print(reduced_evaluate_solution(solution, car_dist_arrival, post_eval=True, seq_decisions=True), solution)