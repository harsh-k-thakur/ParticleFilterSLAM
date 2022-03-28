#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import math
import yaml


# In[3]:


from pr2_utils import *
from data_initialization import *


# In[ ]:


if __name__ == '__main__':
    print("Prediction Step Python File")


# In[ ]:


'''
This function is used to predict the next step of the model
Depending on the next step
We will try to rearange things

Input: Position of the Particle/Particles
Output: Updated position of the Particles using the motion model
'''
def prediction_step(mu, robot=False):
    # Noise Term
    # Activate when you are ready for the whole dataset
    # This normal(mean, standard_deviation, number_of_values)
    # print(mu)
    
    # noise = np.array([[0, 0, 0]])
    # print(noise)
    
    # Global Variables
    global global_encoder_previous_time_stamp
    global global_encoder_current_time_stamp
    global global_fog_previous_time_stamp
    global global_fog_current_time_stamp
    global robot_path
    
    # Function
    global_encoder_current_time_stamp = global_encoder_previous_time_stamp + 1
    global_fog_current_time_stamp = global_fog_previous_time_stamp + 1    
    
    # print("Encoder previous: ", global_encoder_previous_time_stamp)
    # print("Encoder current: ", global_encoder_current_time_stamp)
    
    # Keep track of timestamp of encoder which is used as tau
    previous_encoder_time = encoder_time_stamp[global_encoder_previous_time_stamp]
    current_encoder_time = encoder_time_stamp[global_encoder_current_time_stamp]
    
    # print()
    # print("FOG previous: ", global_fog_previous_time_stamp)   
    # Keep track of timestamp of fog which is used to get delta_yaw
    previous_fog_time = fog_time_stamp[global_fog_previous_time_stamp]
    while True:
        current_fog_time = fog_time_stamp[global_fog_current_time_stamp]
        if approximately_equal(current_fog_time, current_encoder_time):
            break
        else:
            global_fog_current_time_stamp = global_fog_current_time_stamp + 1    
    # print("FOG current: ", global_fog_current_time_stamp)      
    
    # Velocity of Wheel
    encoder_left_wheel_data = wheel_count[global_encoder_current_time_stamp, 0] - wheel_count[global_encoder_previous_time_stamp, 0]
    encoder_right_wheel_data = wheel_count[global_encoder_current_time_stamp, 1] - wheel_count[global_encoder_previous_time_stamp, 1]
    tau = current_encoder_time - previous_encoder_time
    vl = (np.pi * encoder_left_diameter * encoder_left_wheel_data) / (encoder_resolution*tau)
    vr = (np.pi * encoder_right_diameter * encoder_right_wheel_data) / (encoder_resolution*tau)
    v = (vl + vr) / 2
    
    # Theta_t of Fog
    delta_yaw = sum(fog_yaw[global_fog_previous_time_stamp:global_fog_current_time_stamp])
    theta_t = mu[:, 2]
    
    # Adding the noise term to omega
    # Mean=0 and Varince=1e-12
    # noise = np.random.normal(0, 0, theta_t.shape)
    noise_x = np.random.normal(0, 1e-8, theta_t.shape)
    noise_y = np.random.normal(0, 1e-8, theta_t.shape)
    
    # Omega for the angular velocity
    omega = np.full(theta_t.shape, (delta_yaw) / tau)
    
    # X_current is the current position of the robot
    x_current = mu
    x = v*np.cos(theta_t) + noise_x
    y = v*np.sin(theta_t) + noise_y
    omega = omega
    
    # Convert to vectorize form
    se = np.transpose(np.vstack([x, y, omega]))
    sv = tau * se
    
    x_next = x_current + sv
    
    # This is helpful just to create the trajectory
    # Can be commented later in order to save time
    # ex = float(x_next[:, 0])*math.cos(omega)
    # ey = float(x_next[:, 1])*math.sin(omega)
    # ez = 0
    # robot_path.append([[ex, ey, ez]])
    if robot == True:
        robot_path.append(x_next.tolist())
    
    # Once I have x_next I will make the theta_t+1 = 0 and convert it into world frame.
    # This might be helpful for plotting things
    '''
    print()
    print("Velocity: ", v*1e+9)    
    print("Theta: ", theta_t)
    print("Omega: ", omega*1e+9)
    
    # print()
    # print("Tau: ", tau)
    # print("Delta T: ", delta_t)
    
    print()
    print(x_current)  
    print(sv)    
    print(x_next)
    '''
    
    global_encoder_previous_time_stamp = global_encoder_current_time_stamp 
    global_fog_previous_time_stamp = global_fog_current_time_stamp    
    return x_next


# In[ ]:




