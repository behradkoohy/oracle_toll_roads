from collections import Counter, deque, defaultdict
from functools import partial

import numpy as np
from numpy import mean, quantile, median
from numpy import min as nmin
from numpy import max as nmax


timesteps = 30


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


def alternative_get_new_travel_times(
    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, time_out_car_partial_solution
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
        """
        OLD APPROACH
        """
        # # Worst case travel time
        # best_potential_eta = max_travel_eta
        # # as a precaution, set the travel time to the maximum value
        # # new_travel_times[road] = best_known_travel_time
        # # now find all the cars that are leaving during their trip
        # cars_leaving_during_trip = {
        #     ti: time_out_car[road][ti]
        #     for ti in range(time + 1, round(max_travel_eta) + 1)
        #     if time_out_car[road][ti] > 0
        # }
        # total_cars_left_until_t = 0
        # for x, n_vehicles in cars_leaving_during_trip.items():
        #     total_cars_left_until_t += n_vehicles
        #     # calculate what the ETA is if they left later on in the simulation
        #     # We don't want to use x, we want the offset (timesteps after x, i.e. how long you will wait) for the travel time
        #     potential_eta = time + (x - time) + road_vdf(len(new_road_queues[road]) - total_cars_left_until_t)
        #     if potential_eta < best_potential_eta:
        #         best_potential_eta = potential_eta
        #         best_known_travel_time = best_potential_eta - time
        # new_travel_times[road] = best_known_travel_time
        """
        NEW APPROACH
        """
        cars_on_road = len(new_road_queues[road])
        # NOTE: This is taking up 88.5% of our time
        # cars_leaving_during_trip_old = {
        #     ti: time_out_car[road][ti]
        #     for ti in range(time + 1, round(max_travel_eta) + 1)
        #     if time_out_car[road][ti] > 0
        # }

        # remove the current time from the partial solution if it exists
        time_out_car_partial_solution[road].pop(time, None)
        # next, we need to know what the max key in the partial soln is
        current_max_partial_sol = time if len(time_out_car_partial_solution[road]) == 0 else max(time_out_car_partial_solution[road])
        # if the max is greater than the max of the search range, we compute up to the range
        # print(current_max_partial_sol, round(max_travel_eta))
        if current_max_partial_sol < round(max_travel_eta):
            for ti in range(current_max_partial_sol + 1, round(max_travel_eta) + 1):
                if time_out_car[road][ti] > 0:
                    time_out_car_partial_solution[road][ti] = time_out_car[road][ti]
        # if the max is lesser than the max of the search range, we can drop the irrelevant values
        elif current_max_partial_sol > round(max_travel_eta):
            for ti in range(round(max_travel_eta) + 1, current_max_partial_sol + 1):
                time_out_car_partial_solution[road].pop(time, None)
        # set cars_leaving_during_trip to the road
        cars_leaving_during_trip = time_out_car_partial_solution[road]

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
        # print(time, road, cars_leaving_during_trip, best_time_out)
        new_travel_times[road] = best_time_out

        # vehicles_on_road = len(new_road_queues[road])
        # road_vdf = roadVDFS[road]
        # max_time_out = time + (road_vdf(vehicles_on_road))
        # total_v_out_until_x = 0
        # maintained_time_out = max_time_out
        # x_at_maint_time_out = time
        # new_travel_times[road] = max_time_out - time
        # # print(max_time_out - x_at_maint_time_out)
        # cars_leaving_during_trip = {
        #     ti: time_out_car[road][ti]
        #     for ti in range(time + 1, round(max_time_out) + 1)
        #     if time_out_car[road][ti] > 0
        # }
        # cars_out_in_range = sum(cars_leaving_during_trip.values())
        # if cars_out_in_range == 0:
        #     new_travel_times[road] = max_time_out - time
        # else:
        #     for x, n_vehicles in cars_leaving_during_trip.items():
        #         total_v_out_until_x += n_vehicles
        #         travel_time_at_x = road_vdf(vehicles_on_road - total_v_out_until_x)
        #         time_out_at_x = x + travel_time_at_x
        #         # print((x-time), max_time_out, maintained_time_out, travel_time_at_x, time_out_at_x)
        #         # print(max_time_out - time, time_out_at_x - time)
        #         if maintained_time_out > (time_out_at_x - time):
        #             # NOTE: There is something wrong here
        #             # TODO: find the cause of the bug that causes difference in travel time between
        #             # TODO: timestep 10 and 11. The travel time shouldn't increase so drastically
        #             # TODO: because of one car.
        #             # print("old time out:", max_time_out, "new time out", (time_out_at_x - time))
        #             maintained_time_out = time_out_at_x - time
        #             x_at_maint_time_out = x
        #             new_travel_times[road] = time_out_at_x - time
        # if time_out_at_x < maintained_time_out:
        #     maintained_time_out = time_out_at_x
        #     x_at_maint_time_out = x
        #     new_travel_times[road] = time_out_at_x - x
    return new_travel_times, new_road_queues, arrived_vehicles, time_out_car_partial_solution


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
    time_out_car_partial_solution = {r: {time + 1: 0} for r in roadVDFS.keys()}
    arrival_timestep_dict = Counter(car_dist_arrival)
    sol_deque = deque(solution)
    while not is_simulation_complete(roadQueues, time):
        # get the new vehicles at this timestep

        roadTravelTime, roadQueues, arrived_vehicles, time_out_car_partial_solution = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, time_out_car_partial_solution
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
                roadTravelTime, roadQueues, arrived_vehicles, time_out_car_partial_solution = alternative_get_new_travel_times(
                    roadQueues, roadVDFS, time, arrived_vehicles, time_out_car, time_out_car_partial_solution
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
