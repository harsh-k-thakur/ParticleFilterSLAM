# ParticleFilterSLAM

In this project, I have implemented Particle Filter for mapping the free space of the world. 

The data I have used for this is available on my [google drive](https://drive.google.com/drive/folders/1eINhx5H7ci_XB-raPvW9tla8Qe6qYZjG?usp=sharing). 
There are multiple data that is used to create this peoject. 

  1. IMU 
  2. Encoder
  3. LiDAR 
  4. Stereo

And as this is a robotics project, the rotation and transformation of the each particle are given in the [param](/data/param/) folder of this repository.
This would help to locate the position of different sensor on my autonomous car.

The report for this project is available [here](/Report/Particle%20Filter%20SLAM%20for%20creating%20the%20MAP.pdf).

The main code is available [here](/code/partile_filter_slam.ipynb).
For this project I have heavily used Python and rest created everything from scratch.
