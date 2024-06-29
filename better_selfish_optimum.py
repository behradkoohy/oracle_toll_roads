from collections import Counter, defaultdict
from functools import partial

import numpy.random as nprand
from utils import (
    volume_delay_function,
    evaluate_solution,
    is_simulation_complete,
    get_new_travel_times,
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


while not is_simulation_complete(roadQueues, time):
    num_vehicles_arrives = arrival_timestep_dict[time]
    if num_vehicles_arrives == 0:
        # Here we want to just update the times for both roads, removing any vehicles which have already arrived.
        roadTravelTime, roadQueues, arrived_vehicles = get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )

    else:
        # I think first we have to remove any arrived vehicles because otherwise it won't be done
        for road in roadQueues.keys():
            arrived_vehicles = arrived_vehicles + [
                car for car in roadQueues[road] if car[3] <= time
            ]
            roadQueues[road] = [car for car in roadQueues[road] if car[3] > time]
            # NOTE: THIS BELOW IS OLD AND NEEDS REPLACING IF ITS IMPORTANT
            # vehicles_on_road = len(roadQueues[road])
            # road_vdf = roadVDFS[road]
            # max_time_out = time + (road_vdf(vehicles_on_road))
            # total_v_out_until_x = 0
            # maintained_time_out = max_time_out
            # cars_out_in_range = sum([time_out_car[road][ti] for ti in range(time + 1, round(max_time_out) + 1)])
            # for x in range(time + 1, round(max_time_out) + 1):
            #     vehicles_out_at_timestep_x = time_out_car[road][x]
            #     if vehicles_out_at_timestep_x == 0:
            #         continue
            #
            #     total_v_out_until_x += vehicles_out_at_timestep_x
            #     travel_time_at_x = (road_vdf(vehicles_on_road - total_v_out_until_x))
            #     time_out_at_x = x + travel_time_at_x
            #     # print(x, time_out_at_x, maintained_time_out)
            #     if time_out_at_x < maintained_time_out:
            #         maintained_time_out = time_out_at_x
            #         roadTravelTime[road] = time_out_at_x - x
            #
            #     if cars_out_in_range == vehicles_out_at_timestep_x:
            #         break
            # if total_v_out_until_x == 0:
            #     roadTravelTime[road] = max_time_out - time
        # Here we will go through and add cars to the best option, and then we'll update the travel time for the next car
        for vehicle in range(num_vehicles_arrives):
            if roadTravelTime[1] < roadTravelTime[2]:
                quickest_road, travel_time = 1, roadTravelTime[1]
                slow_road, slow_tt = 2, roadTravelTime[2]
            else:
                quickest_road, travel_time = 2, roadTravelTime[2]
                slow_road, slow_tt = 1, roadTravelTime[1]
            roadQueues[quickest_road] = roadQueues[quickest_road] + [
                (quickest_road, time, time + travel_time, time + travel_time)
            ]
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

            solution.append(quickest_road)
            time_out_car[quickest_road][round(time + travel_time)] = (
                time_out_car[quickest_road][round(time + travel_time)] + 1
            )
            # print("Old road travel time", roadTravelTime, "with queues", {r: len(x) for r,x in roadQueues.items()}, quickest_road)
            roadTravelTime, roadQueues, arrived_vehicles = get_new_travel_times(
                roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
            )
            # print("New road travel time", roadTravelTime, "with queues", {r: len(x) for r, x in roadQueues.items()})

    time += 1

print(len(solution), len(car_dist_arrival), len(arrived_vehicles))
print(evaluate_solution(solution, car_dist_arrival, post_eval=True), solution)
