from collections import Counter, defaultdict, deque
from functools import partial
import numpy.random as nprand
from numpy import mean, quantile, median, max
from numpy import min as nmin

from utils import (
    is_simulation_complete,
    volume_delay_function,
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


# solution = [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1,
#  2, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2,
#  2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1,
#  2, 1, 1, 1,]
solution = [2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2]


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


sol_deque = deque(solution)


while not is_simulation_complete(roadQueues, time):
    roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
        roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
    )
    num_vehicles_arrived = arrival_timestep_dict[time]
    if num_vehicles_arrived is None:
        num_vehicles_arrived = 0
    decisions = [sol_deque.popleft() for _ in range(num_vehicles_arrived)]
    # add vehicles to the new queue
    for decision, n in zip(decisions, range(num_vehicles_arrived)):
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
        # print(roadTravelTime, {r: len(x) for r, x in roadQueues.items()})
        print(
            "I am vehicle",
            n,
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
        roadTravelTime, roadQueues, arrived_vehicles = alternative_get_new_travel_times(
            roadQueues, roadVDFS, time, arrived_vehicles, time_out_car
        )
    time += 1
    # if time == 13:
    #     exit()

travel_time = [c[3] - c[1] for c in arrived_vehicles]
print([(c[1], c[3]) for c in arrived_vehicles])
print(mean(travel_time))
# if post_eval:
print(
    nmin(travel_time),
    quantile(travel_time, 0.25),
    mean(travel_time),
    median(travel_time),
    quantile(travel_time, 0.75),
    max(travel_time),
)
