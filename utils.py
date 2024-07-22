from collections import Counter, deque, defaultdict
from functools import partial

import json


import numpy as np
from numpy import mean, quantile, median
from numpy import min as nmin
from numpy import max as nmax


timesteps = 30


def write_solutions_to_file(results_dict, filename='results.json'):
    with open (filename, 'w') as f:
        json.dump(results_dict, f)

def is_simulation_complete(roadQueues, time):
    if time < timesteps:
        return False
    if all([len(r) == 0 for r in roadQueues.values()]):
        # print("terminating early")
        # print([len(r) == 0 for r in roadQueues.values()], all([len(r) == 0 for r in roadQueues.values()]))
        return True
    else:
        return False


def volume_delay_function(a, b, c, t0, v):
    """
    :param v: volume of cars on road currently
    :param a: VDF calibration parameter alpha, values of 0.15 default
    :param b: VDF calibration parameter beta, value of 4 default
    :param c: road capacity
    :param t0: free flow travel time
    :return: travel time for a vehicle on road

    link for default value info:
    https://www.degruyter.com/document/doi/10.1515/eng-2022-0022/html?lang=en
    """
    # a = 0.15
    a = 0.656
    b = 4.8
    # b = 4
    # c = 30
    # t0 = 30
    return t0 * (1 + (a * pow((v / c), b)))


def get_new_travel_times(roadQueues, roadVDFS, time, arrived_vehicles, time_out_car):
    new_travel_times = {}
    new_road_queues = {}
    for road in roadQueues.keys():
        arrived_vehicles = arrived_vehicles + [
            car for car in roadQueues[road] if car[3] <= time
        ]
        new_road_queues[road] = [car for car in roadQueues[road] if car[3] > time]
        vehicles_on_road = len(new_road_queues[road])
        road_vdf = roadVDFS[road]
        max_time_out = time + (road_vdf(vehicles_on_road))
        total_v_out_until_x = 0
        maintained_time_out = max_time_out
        x_at_maint_time_out = time
        cars_out_in_range = sum(
            [time_out_car[road][ti] for ti in range(time + 1, round(max_time_out) + 1)]
        )
        # smth = {ti: time_out_car[road][ti] for ti in range(time + 1, round(max_time_out) + 1)}
        if cars_out_in_range == 0:
            new_travel_times[road] = max_time_out - time
        else:
            for x in range(time + 1, round(max_time_out)):
                vehicles_out_at_timestep_x = time_out_car[road][x]
                if vehicles_out_at_timestep_x == 0:
                    continue
                total_v_out_until_x += vehicles_out_at_timestep_x
                travel_time_at_x = road_vdf(vehicles_on_road - total_v_out_until_x)
                time_out_at_x = x + travel_time_at_x
                if time_out_at_x < maintained_time_out:
                    maintained_time_out = time_out_at_x
                    x_at_maint_time_out = x
                    new_travel_times[road] = time_out_at_x - x
                if cars_out_in_range == vehicles_out_at_timestep_x:
                    new_travel_times[road] = maintained_time_out - x_at_maint_time_out
                    break
        if road not in new_travel_times:
            new_travel_times[road] = max_time_out - time
    return new_travel_times, new_road_queues, arrived_vehicles

def get_cars_leaving_during_trip(time_out_car, road, time, max_travel_eta):
    road_dict = time_out_car[road]  # Pre-fetch the dictionary for the specific road
    end_time = round(max_travel_eta) + 1  # Calculate range endpoint once
    timesteps_to_check = [ti for ti in range(time + 1, round(end_time) + 1) if road_dict[ti] > 0]
    return {ti: road_dict[ti] for ti in timesteps_to_check}

def cars_out_range_generator(time, max_travel_eta, road_dict):
    n = time
    while n <= round(max_travel_eta):
        n = n + 1
        if n in road_dict:
            yield n, road_dict[n]


def generate_new_road_travel_time(time_out_car, road, time, max_travel_eta, road_vdf, cars_on_road):
    road_dict = time_out_car[road]
    best_road_travel_time = max_travel_eta - time
    cum_cars_left_at_t = 0
    for future_timestep, cars_left in cars_out_range_generator(time, max_travel_eta, road_dict):
        cum_cars_left_at_t += cars_left
        time_offset = future_timestep - time
        travel_time_at_fut_time = time_offset + road_vdf(cars_on_road - cum_cars_left_at_t)
        # if best_road_travel_time < travel_time_at_fut_time:
        #     return best_road_travel_time
        if best_road_travel_time > travel_time_at_fut_time:
            best_road_travel_time = travel_time_at_fut_time
    return best_road_travel_time

def alternative_get_new_travel_times(
    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
):
    new_travel_times = {}
    new_road_queues = {}
    for road in roadQueues.keys():
        arrived_vehicles = arrived_vehicles + [
            car for car in roadQueues[road] if car[3] <= time
        ]
        new_road_queues[road] = [car for car in roadQueues[road] if car[3] > time]
        road_vdf = roadVDFS[road]
        best_known_travel_time = road_vdf(len(new_road_queues[road]))
        max_travel_eta = time + best_known_travel_time
        cars_on_road = len(new_road_queues[road])
        # NOTE: This is taking up 88.5% of our time
        """cars_leaving_during_trip = {
            ti: time_out_car[road][ti]
            for ti in range(time + 1, round(max_travel_eta) + 1)
            if time_out_car[road][ti] > 0
        }"""
        # NOTE: this now has to completely change as time_out_car is no longer a defaultdict -> just a dict now
        cars_leaving_during_trip = get_cars_leaving_during_trip(time_out_car, road, time, max_travel_eta)
        cumsum_base = 0
        cars_leaving_cumsum = [
            cumsum_base := cumsum_base + n for n in cars_leaving_during_trip.values()
        ]
        cars_leaving_during_trip_sum = {time: 0} | {
            ti: cars
            for ti, cars in zip(cars_leaving_during_trip.keys(), cars_leaving_cumsum)
        }
        # print(road, cars_leaving_during_trip)
        cars_leaving_during_trip_new_tt = {
            ti: ti + road_vdf(cars_on_road - cars_out) - time
            for ti, cars_out in cars_leaving_during_trip_sum.items()
        }
        best_time_out = min(cars_leaving_during_trip_new_tt.values())
        # best_time_out = generate_new_road_travel_time(time_out_car, road, time, max_travel_eta, road_vdf, cars_on_road)
        # print(time, road, cars_leaving_during_trip, best_time_out)
        new_travel_times[road] = best_time_out
    return new_travel_times, new_road_queues, arrived_vehicles


def evaluate_solution(solution, car_dist_arrival, post_eval=False, seq_decisions=False):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """
    if len(solution) != len(car_dist_arrival):
        raise Exception("Length of solution and car_dist_arrival must be equal")
    roadQueues = {r: [] for r in set(solution)}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}
    arrived_vehicles = []
    time = 0
    arrival_timestep_dict = Counter(car_dist_arrival)
    sol_deque = deque(solution)
    while not is_simulation_complete(roadQueues, time):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )

        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        if num_vehicles_arrived is None:
            num_vehicles_arrived = 0
        # just collect the decision of the vehicles
        decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
        # add vehicles to the new queue
        for decision in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    time + roadTravelTime[decision],
                    time + roadTravelTime[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                (
                    roadTravelTime,
                    roadQueues,
                    arrived_vehicles,
                ) = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                )

        time += 1

    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    # print([(c[3], c[1], c[3] - c[1]) for c in arrived_vehicles])
    # print([c for c in arrived_vehicles if c[0] == 2])
    # travel_time = np.asarray(travel_time, dtype=np.float64)
    if post_eval:
        print(
            nmin(travel_time),
            quantile(travel_time, 0.25),
            mean(travel_time),
            median(travel_time),
            quantile(travel_time, 0.75),
            nmax(travel_time),
        )
    return -mean(travel_time)


def reduced_is_simulation_complete(roadQueues, time):
    if time >= timesteps + 1:
        return True
    else:
        return False


def reduced_evaluate_solution(
    solution, car_dist_arrival, post_eval=False, seq_decisions=False
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """
    if len(solution) != len(car_dist_arrival):
        raise Exception("Length of solution and car_dist_arrival must be equal")
    roadQueues = {r: [] for r in set(solution)}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    time_out_car = {r: defaultdict(int) for r in roadVDFS.keys()}
    arrived_vehicles = []
    time = 0
    arrival_timestep_dict = Counter(car_dist_arrival)
    sol_deque = deque(solution)
    while not reduced_is_simulation_complete(roadQueues, time):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )

        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        if num_vehicles_arrived is None:
            num_vehicles_arrived = 0
        # just collect the decision of the vehicles
        decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
        # add vehicles to the new queue
        for decision in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    time + roadTravelTime[decision],
                    time + roadTravelTime[decision],
                )
            ]
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            )
            if seq_decisions:
                (
                    roadTravelTime,
                    roadQueues,
                    arrived_vehicles,
                ) = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                )

        time += 1
    # Anything which is still in road queue can be added to arrived vehicles
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]

    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    # print([(c[3], c[1], c[3] - c[1]) for c in arrived_vehicles])
    # print([c for c in arrived_vehicles if c[0] == 2])
    # travel_time = np.asarray(travel_time, dtype=np.float64)
    if post_eval:
        print(
            nmin(travel_time),
            quantile(travel_time, 0.25),
            mean(travel_time),
            median(travel_time),
            quantile(travel_time, 0.75),
            nmax(travel_time),
        )
    return -mean(travel_time)

def gen_evaluate_solution(
    solution, car_dist_arrival, post_eval=False, seq_decisions=False
):
    """
    solution: list of roads for the vehicles to take, i.e. [1,1,2,2,1,1,2,....,]
    car_dist_arrival: list of arrival times of vehicles, length n, i.e. [1,1,2,3,...,30]
    """
    if len(solution) != len(car_dist_arrival):
        raise Exception("Length of solution and car_dist_arrival must be equal")
    roadQueues = {r: [] for r in set(solution)}
    roadVDFS = {
        1: partial(volume_delay_function, 0.656, 4.8, 15, 20),
        2: partial(volume_delay_function, 0.656, 4.8, 30, 20),
    }
    roadTravelTime = {r: roadVDFS[r](0) for r in roadVDFS.keys()}
    time_out_car = {r: {} for r in roadVDFS.keys()}
    arrived_vehicles = []
    time = 0
    arrival_timestep_dict = Counter(car_dist_arrival)
    sol_deque = deque(solution)
    while not reduced_is_simulation_complete(roadQueues, time):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )

        # Add new vehicles from here
        num_vehicles_arrived = arrival_timestep_dict[time]
        if num_vehicles_arrived is None:
            num_vehicles_arrived = 0
        # just collect the decision of the vehicles
        decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
        # add vehicles to the new queue
        for decision in decisions:
            roadQueues[decision] = roadQueues[decision] + [
                (
                    decision,
                    time,
                    time + roadTravelTime[decision],
                    time + roadTravelTime[decision],
                )
            ]
            # time_out_car[decision][round(time + roadTravelTime[decision])] = (
            #     time_out_car[decision][round(time + roadTravelTime[decision])] + 1
            # )
            time_out_car[decision][round(time + roadTravelTime[decision])] = (
                time_out_car[decision][round(time + roadTravelTime[decision]), 0] + 1
            )


            if seq_decisions:
                (
                    roadTravelTime,
                    roadQueues,
                    arrived_vehicles,
                ) = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
                )

        time += 1
    # Anything which is still in road queue can be added to arrived vehicles
    for road, queue in roadQueues.items():
        arrived_vehicles = arrived_vehicles + [car for car in roadQueues[road]]

    travel_time = [c[3] - c[1] for c in arrived_vehicles]
    # print([(c[3], c[1], c[3] - c[1]) for c in arrived_vehicles])
    # print([c for c in arrived_vehicles if c[0] == 2])
    # travel_time = np.asarray(travel_time, dtype=np.float64)
    if post_eval:
        print(
            nmin(travel_time),
            quantile(travel_time, 0.25),
            mean(travel_time),
            median(travel_time),
            quantile(travel_time, 0.75),
            nmax(travel_time),
        )
    return -mean(travel_time)