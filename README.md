# racerbot.rl
autonomous racecar minimizing lap time using reinforcement learning  
https://github.com/acm-uiuc/racerbot.rl  
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

# Resources
https://arxiv.org/pdf/1710.02410.pdf  
https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf  
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html  
https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53  
https://docs.google.com/document/d/1qA9CaBvesFW-PzkKNyNZu4MHOpyo6yaOSsx0UsCUAUE/edit  
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
http://planning.cs.uiuc.edu/node658.html  





