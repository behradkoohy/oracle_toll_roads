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
    alternative_get_new_travel_times,
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

# def get_new_travel_times(roadQueues, roadVDFS, time, arrived_vehicles, time_out_car):
#     new_travel_times = {}
#     new_road_queues = {}
#     for road in roadQueues.keys():
#         arrived_vehicles = arrived_vehicles + [
#             car for car in roadQueues[road] if car[3] <= time
#         ]
#         new_road_queues[road] = [car for car in roadQueues[road] if car[3] > time]
#         vehicles_on_road = len(new_road_queues[road])
#         road_vdf = roadVDFS[road]
#         max_time_out = time + (road_vdf(vehicles_on_road))
#         total_v_out_until_x = 0
#         maintained_time_out = max_time_out
#         x_at_maint_time_out = time
#         cars_out_in_range = sum([time_out_car[road][ti] for ti in range(time + 1, round(max_time_out) + 1)])
#         if cars_out_in_range == 0:
#             new_travel_times[road] = max_time_out - time
#         else:
#             for x in range(time + 1, round(max_time_out)):
#                 vehicles_out_at_timestep_x = time_out_car[road][x]
#                 if vehicles_out_at_timestep_x == 0:
#                     continue
#                 total_v_out_until_x += vehicles_out_at_timestep_x
#                 travel_time_at_x = (road_vdf(vehicles_on_road - total_v_out_until_x))
#                 time_out_at_x = x + travel_time_at_x
#                 if time_out_at_x < maintained_time_out:
#                     maintained_time_out = time_out_at_x
#                     x_at_maint_time_out = x
#                     new_travel_times[road] = time_out_at_x - x
#                 if cars_out_in_range == vehicles_out_at_timestep_x:
#                     new_travel_times[road] = maintained_time_out - x_at_maint_time_out
#                     break
#         if road not in new_travel_times:
#             new_travel_times[road] = max_time_out - time
#     print("BELOW THIS", new_travel_times, {r: len(x) for r, x in new_road_queues.items()})
#     return new_travel_times, new_road_queues, arrived_vehicles


while not is_simulation_complete(roadQueues, time):
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
        print(
            "I am vehicle",
            vehicle,
            ". I am choosing",
            quickest_road,
            "at time",
            time,
            "with travel time",
            travel_time,
            "and ETA of",
            time + travel_time,
            ". The alternative is",
            slow_road,
            "with travel time",
            slow_tt,
            "and ETA of",
            time + slow_tt,
        )
        # print({r:len(x) for r,x in roadQueues.items()})
        solution.append(quickest_road)
        time_out_car[quickest_road][round(time + travel_time)] = (
            time_out_car[quickest_road][round(time + travel_time)] + 1
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
# print([(c[1], c[3]) for c in arrived_vehicles])
print(len(solution), len(car_dist_arrival), len(arrived_vehicles))
print(evaluate_solution(solution, car_dist_arrival, post_eval=True, seq_decisions=True), solution)