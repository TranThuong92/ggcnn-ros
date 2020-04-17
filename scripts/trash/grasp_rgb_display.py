#!/usr/bin/env python

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
from ggcnn_pred import predict, process_output, processing_depth
from ggcnn.msg import grasp_point

VERBOSE = False

# Calibration parameter
cam_fx = 606
cam_fy = 316
cam_tx = 605
cam_ty = 245

class grasp_robot:
    def __init__(self):
        #self.grasp_pub = rospy.Publisher("grasp_config_robot", grasp_poin  t, queue_size=3)
        self.color_img = rospy.Subscriber("/camera/color/image_raw", Image, self.callback_dis)
        self.grasp_pred = rospy.Subscriber("grasp_config", grasp_point, self.callback)
        self.msg = grasp_point()
        self.px = 0
        self.py = 0
        self.pz = 0
        self.pt1 = np.zeros(2)
        self.pt2 = np.zeros(2)
        self.pt3 = np.zeros(2)
        self.pt4 = np.zeros(2)
        self.COLOR = (0, 255, 0)

    def callback(self, ros_msg):
        # Invoke the image from camera
        grasp_x = ros_msg.center[0]
        grasp_y = ros_msg.center[1]
        depth = ros_msg.depth

        self.pt1 = np.array(ros_msg.pt1)
        self.pt2 = np.array(ros_msg.pt2)
        self.pt3 = np.array(ros_msg.pt3)
        self.pt4 = np.array(ros_msg.pt4)

        self.pz = depth
        self.px = (grasp_x-cam_tx)*self.pz/cam_fx
        self.py = (grasp_y-cam_ty)*self.pz/cam_fy

        #print(self.px, self.py, self.pz)
        """
        try:
            self.grasp_pub.publish(self.msg)
        except KeyboardInterrupt:
            print('Publish error!')
        """

    def callback_dis(self, ros_msg):
        color_image = CvBridge().imgmsg_to_cv2(ros_msg)
        pts = np.array([self.pt1, self.pt2, self.pt3, self.pt4], np.int32)
        cv2.polylines(color_image, [pts], True, self.COLOR )
        cv2.imshow("Color Image with Grasp", color_image)
        cv2.waitKey(3)

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
