#!/usr/bin/env python
import cv2
import rospy
import message_filters
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

def callback(rgb_msg, camera_info):
    rgb_image = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
    camera_info_K = np.array(camera_info.K).reshape([3, 3])
    camera_info_D = np.array(camera_info.D)
    rgb_undist = cv2.undistort(rgb_image, camera_info_K, camera_info_D)
    print(rgb_image)

def listener():
    rospy.init_node('webcam_listener', anonymous=True)
    image_sub = message_filters.Subscriber('/webcam/image_raw', Image)
    info_sub = message_filters.Subscriber('/webcam/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], 10, 0.2)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
