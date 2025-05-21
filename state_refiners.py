import numpy
from numpy import cos, sin

def acrobot_refiner(raw_state):
    refined_state = numpy.array(
        [cos(raw_state[0]), sin(raw_state[0]), cos(raw_state[1]), sin(raw_state[1]), raw_state[2], raw_state[3]], dtype=numpy.float32
    )
    return refined_state

def cart_pole_refiner(raw_state):
    refined_state = numpy.array(raw_state, dtype=numpy.float32)
    return refined_state

def mountain_car_refiner(raw_state):
    refined_state = numpy.array(raw_state, dtype=numpy.float32)
    return refined_state

def taxi_refiner(raw_state):
    refined_state = int(raw_state)
    return refined_state

def frozen_lake_refiner(raw_state):
    refined_state = int(raw_state)
    return refined_state


refiners = {
    "Acrobot_v1": acrobot_refiner,
    "CartPole_v1": cart_pole_refiner,
    "MountainCar_v0": mountain_car_refiner,
    "Taxi_v3": taxi_refiner,
    "FrozenLake_v1": frozen_lake_refiner
}
