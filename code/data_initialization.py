import numpy as np
import pandas as pd
import cv2
import math
import yaml
from pr2_utils import *

'''
This will help to create an empty map of the world
We will start will small
This is not dynamic, we manually the values.

Input: N length in one direction
Output: NxN Matrix with zeros
'''
def create_map(N, M):
    world_map = np.zeros((N, M), dtype=np.int8)
    return world_map


def initialize_particles(N):
    '''
        There will be two matrix
        1. Initial Particle set mu=[0, 0, 0]T. This will be Nx3 in size [X, Y, Theta]
        2. Initial Weight Vector alpha = [1/N]. This will be Nx1 in size 
    '''
    mu = np.zeros((N, 3), dtype=np.int32)
    alpha = np.full((N, 1), 1/N)
    return mu, alpha


'''
This function is used to load the lidar data
This lidar data is generally present in "data/sensor_data/lidar.csv"
But you pass it as a filename as well

Input :- File Location
Output :- Returns the angle, timestamp, and lidar data
'''
def get_lidar_data(filename="data/sensor_data/lidar.csv"):
    time_stamp, lidar_data = read_data_from_csv(filename)
    angle = np.linspace(-5, 185, 286) / 180 * np.pi
    # print(lidar_data.shape)    
    return angle, time_stamp, lidar_data


'''
This function is used to load the encoder data
This encoder data is generally present in "data\sensor_data\encoder.csv"
But you pass it as a filename as well

Input :- File Location
Output :- Returns the timestamp and wheel count
'''
def get_encoder_data(filename="data/sensor_data/encoder.csv"):
    time_stamp, wheel_count = read_data_from_csv(filename)    
    return time_stamp, wheel_count


'''
This function is used to load the fog data
This encoder data is generally present in "data\sensor_data\fog.csv"
But you pass it as a filename as well

Input :- File Location
Output :- Returns the timestamp and fog_data, only yaw required
'''
def get_fog_data(filename="data/sensor_data/fog.csv"):
    time_stamp, fog_data = read_data_from_csv(filename)
    # print(fog_data[0])
    fog_yaw = fog_data[:, 2]
    
    
    return time_stamp, fog_yaw
    


if __name__ == '__main__':
    print("Data Initialization File")
