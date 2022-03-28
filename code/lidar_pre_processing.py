import numpy as np
import pandas as pd
import timeit
import cv2
import math
import yaml

from pr2_utils import *

v_R_l = np.array([[ 0.00130201,     0.796097,    0.605167],
                  [   0.999999, -0.000419027, -0.00160026],
                  [-0.00102038,     0.605169,   -0.796097]])
        
v_P_l = np.array([0.8349, -0.0126869, 1.76416])

Zero_T = np.zeros((3))


v_T_l = np.array([[ 0.00130201,     0.796097,    0.605167,     0.8349],
                  [   0.999999, -0.000419027, -0.00160026, -0.0126869],
                  [-0.00102038,     0.605169,   -0.796097,    1.76416],
                  [          0,            0,           0,          1]])

l_T_v = np.array([[0.0013,    1.0000,   -0.0010,    0.0134],
                  [0.7961,   -0.0004,    0.6052,   -1.7323],
                  [0.6052,   -0.0016,   -0.7961,    0.8992],
                  [     0,         0,         0,    1.0000]])

def world_T_vehicle(theta, x, y):
    """
    This function is used to convert give th pose of the robot in world fram
    The vechile is somewhere in the world frame
    
    Thus this will help us to get the coordinates of the vehicle in world frame
    Input: theta, x and y 
    Output: w_T_v Pose of the robot 
    """
    # print()
    # print("Theta")
    # print(theta[0, 0])
    
    # print()
    # print("X, Y")
    # print(x[0, 0], y[0, 0])
    w_T_v = []
    for i in range(NUMBER_OF_PARTICLES):
        w_T_v.append([[math.cos(theta[0, i]), -math.sin(theta[0, i]),    0,    x[0, i]],
                      [math.sin(theta[0, i]),  math.cos(theta[0, i]),    0,    y[0, i]],
                      [                    0,                      0,    1,          0],
                      [                    0,                      0,    0,          1]])
        
    '''   
    w_T_v = np.array([[np.cos(theta), -np.sin(theta),    0,    x],
                      [np.sin(theta),  np.cos(theta),    0,    y],
                      [            0,              0,    1,    0],
                      [            0,              0,    0,    1]])
    '''
    w_T_v = np.array(w_T_v)
    
    return w_T_v


def convert_lidar_to_world(NUMBER_OF_PARTICLES, x, y, z, w_T_v):
    '''
    This function is used to convert the co-ordinats
    From Lidar Frame to Vehicle Frame

    Input: Coordinates in LIdar Frame (X, Y, Z)
    Output: Coordinates in the Vehicle Frame (X, Y, Z)
    '''
    
    # x = x.astype(int)
    # y = y.astype(int)
    # z = z.astype(int)
    one = np.ones(x.shape)
    
    # print()
    # print("Inside Function")
    
    s_l = np.array([x, y, z, one])
    # print()
    # print("Lidar Position")
    # print(s_l.shape)
    # print(s_l)  
    
    s_l = np.transpose(s_l)
    # print()
    # print("Lidar Transpose Position")
    # print(s_l.shape)
    # print(s_l) 
        
    s_l_new = []
    for i in range(len(s_l)):
        s_l_new.append(s_l[i].T.tolist())
    s_l_new = np.array(s_l_new)
    # print()
    # print("Lidar Needed Position")
    # print(s_l_new.shape)
    # print(s_l_new) 
    
    updated_v_T_l = np.array([v_T_l])
    updated_v_T_l = np.repeat(updated_v_T_l, repeats=NUMBER_OF_PARTICLES, axis=0)
    # print()
    # print("Pose")
    # print(updated_v_T_l.shape)
    # print(updated_v_T_l)
    
    pose_matrix = np.matmul(w_T_v, updated_v_T_l)
    # print()
    # print("Pose Multiplication")
    # print(pose_matrix.shape)
    # print(pose_matrix)
    
    s_v = np.matmul(pose_matrix, s_l_new)
    # print()
    # print("SV Calulated")
    # print(s_v.shape)
    # print(s_v)
    
    new_x, new_y, new_z = s_v[:, 0, :].T, s_v[:, 1, :].T, s_v[:, 2, :].T
    # print(new_x)
    # print(new_x, new_y, new_z)    
    return new_x, new_y, new_z



def convert_lidar_to_cell_new(alpha, mu, lidar_data, angle, NUMBER_OF_PARTICLES):
    # First of all we will map this lidar in 2D frame
    # Using the angle we will get the approximate
    # Starting:- (x, y) and ending:- (x, y)
    
    # sx, sy = mu[best_approximation, 0], mu[best_approximation, 1]
    sx, sy, sz = np.array([mu[:, 0]]), np.array([mu[:, 1]]), np.array([mu[:, 2]])
    
    # This part is do 
    indValid = np.logical_and((lidar_data < 60),(lidar_data > 0.1))
    lidar_data = lidar_data[indValid]
    angle = angle[indValid]
    
    # xy position in the sensor frame    
    x_lidar = lidar_data*np.cos(angle)
    y_lidar = lidar_data*np.sin(angle)
    zeros   = np.zeros(x_lidar.shape)
    ones    = np.ones(x_lidar.shape)
    
    # Theta of the Lidar
    z_lidar = sz
    # print()
    # print("Theta of Lidar")
    # print(z_lidar)
    
    # Get the Transformation Matrix
    w_T_v = world_T_vehicle(z_lidar, sx, sy)
    # print()
    # print("world_T_robot")
    # print(w_T_v.shape)
    # print(w_T_v)
    
    # convert position in the map frame here 
    Y_Lidar = np.stack((x_lidar, y_lidar, zeros, ones))
    row, col = Y_Lidar.shape    
    # print()
    # print("Y_LIDAR")
    # print(row, col)
    # print(Y_Lidar[:, 0])   
    # print(Y_Lidar[:, 1])  
    
    sx = np.repeat(sx, repeats=col, axis=0)
    sy = np.repeat(sy, repeats=col, axis=0)
    
    # Matrix Transformation    
    lidar_x = np.transpose(np.array([Y_Lidar[0, :]]))
    lidar_x = np.repeat(lidar_x, repeats=NUMBER_OF_PARTICLES, axis=1)
    
    lidar_y = np.transpose(np.array([Y_Lidar[1, :]]))
    lidar_y = np.repeat(lidar_y, repeats=NUMBER_OF_PARTICLES, axis=1)
    
    # print()
    # print("Lidar_x, Lidar_y ")
    # print(lidar_x.shape, lidar_y.shape)
    # print(lidar_x, lidar_y)
    
    ex = lidar_x
    ey = lidar_y
    ez = np.zeros(ex.shape)
    # print()
    # print("Ex, EY")
    # print(ex[0], ey[0])
    
    vehicle_x, vehicle_y, vehicle_z = convert_lidar_to_world(NUMBER_OF_PARTICLES, ex, ey, ez, w_T_v)
    # print()
    # print("vehicle_x, vehicle_x ")
    # print(vehicle_x.shape, vehicle_y.shape)
    # print(vehicle_x[0], vehicle_y[0])
    
    Y_World = np.stack((vehicle_x, vehicle_y))
    # print()
    # print("Y_World")
    # print(Y_World.shape)
    # print(Y_World)
        
    return Y_World

if __name__ == '__main__':
    print("Lidar Pre-Processing File")