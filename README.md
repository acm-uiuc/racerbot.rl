# racerbot.rl
autonomous racecar minimizing lap time using reinforcement learning  
<Author: Kazuki Shin>

# Scripts
<preprocess.py>
- mp4 video to frames
- data parsing (parse out crash footage)
- data augmenting (flip, rotate, etc)  
<dataloader.py>
- convert raw data into PyTorch tensors
- enumerate through dataloader in training  
<train.py>
- trains the inputs through custom CNN model
- initial normalization of pixel values (0-255 -> 0-1)
- 5 convolution layers for feature extraction
- 3 fully connected layers for steering angle classification

# Conda Env
conda create -n racecar  
conda activate racecar  
conda install python  

# ROS

Custom ROS package:
- created by Kazuki Shin
- imitation/config/racecar

ROS Dependencies:

ROS Sensor Nodes:
- Logitec USB Camera
- ZED 3D Structure Camera
- IMU Sensor
- mmWave Radar
- VESC motor controller

# CARLA Simulator





