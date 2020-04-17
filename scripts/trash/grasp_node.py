#!/usr/bin/env python


"""
This code is used for old version of ROS Node.
Don't use it!
"""
from __future__ import print_function

import sys
import rospy

# Python libs
import sys, time
# numpy and scipy
import numpy as np
from scipy.ndimage import filters
import cv2
import matplotlib.pyplot as plt
# ROS libraries
import roslib
import rospy
from imageio import imread
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ggcnn.msg import grasp_point

VERBOSE = False

# Calibration parameter
cam_fx = 606
cam_fy = 605
cam_tx = 245
cam_ty = 316

class grasp_robot:
    def __init__(self):
        #self.grasp_pub = rospy.Publisher("grasp_config_robot", grasp_poin  t, queue_size=3)
        self.grasp_pred = rospy.Subscriber("grasp_config", grasp_point, self.callback)
        self.msg = grasp_point()
        self.px = 0
        self.py = 0
        self.pz = 0

    def callback(self, ros_msg):
        # Invoke the image from camera
        grasp_x = ros_msg.center[0]
        grasp_y = ros_msg.center[1]
        depth = ros_msg.depth

        self.pz = depth
        self.px = (grasp_x-cam_tx)*self.pz/cam_fx
        self.py = (grasp_y-cam_ty)*self.pz/cam_fy

        print(self.px, self.py, self.pz)
        print('Center is: ', grasp_x, grasp_y)
        print('Depth is: ', depth)
        """
        try:
            self.grasp_pub.publish(self.msg)
        except KeyboardInterrupt:
            print('Publish error!')
        """

def main(args):
    rospy.init_node('grasp_config_robot', anonymous=True)
    grasp_class = grasp_robot()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down grasp prediction!")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
