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
import timeit
# ROS libraries
import roslib
import rospy
from imageio import imread
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ggcnn_pred import predict, process_output, processing_depth
from ggcnn.msg import grasp_point

VERBOSE = False

depth_read = imread("pcd0100d.tiff")

class grasp_prediction:
    def __init__(self):
        self.grasp_pub = rospy.Publisher("grasp_config", grasp_point, queue_size=3)
        self.bridge = CvBridge()
        self.grasp_pred = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",
                                            Image, self.callback, queue_size=1)
        self.msg = grasp_point()

    def callback(self, ros_msg):
        start_time = time.time()
        # Invoke the image from camera
        ori_image = CvBridge().imgmsg_to_cv2(ros_msg)
        use_image = ori_image.copy()
        use_image = use_image[(use_image.shape[0]-300)/2:(use_image.shape[0]+300)/2,
                            (use_image.shape[1]-300)/2:(use_image.shape[1]+300)/2]
        # Remove black points
        ave_inten = use_image.mean().copy()
        for i in range(use_image.shape[0]):
            for j in range(use_image.shape[1]):
                if use_image[i, j] < 10:
                    use_image[i, j] = ave_inten
        cv_image = use_image/1000.0
        cv_image = np.clip(cv_image, 0.2, 1)
        cv_image = np.uint8(cv_image*255)
        crop_h = int((ori_image.shape[0]-300)/2)
        crop_w = int((ori_image.shape[1]-300)/2)
        offsets = np.array([crop_w, crop_h])
        # Process the image
        depth_img = processing_depth(cv_image)
        #depth_img = processing_depth(depth_read)
        # Predict the grasp
        q_img, cos_img, sin_img, width_img = predict(depth_img)
        cvt_img = np.uint8(q_img*255)
        cvt_img = cv2.applyColorMap(cvt_img, cv2.COLORMAP_JET)
        # Post process the output
        dis_img, grasp_pts, center = process_output(cvt_img, q_img, cos_img, sin_img, width_img, grip_length=30)
        # Publish the grasp points
        center = center + offsets
        self.msg.depth  = ori_image[290, 300]
        self.msg.center = list(center)
        self.msg.pt1    = list(grasp_pts[0]+offsets)
        self.msg.pt2    = list(grasp_pts[1]+offsets)
        self.msg.pt3    = list(grasp_pts[2]+offsets)
        self.msg.pt4    = list(grasp_pts[3]+offsets)
        try:
            self.grasp_pub.publish(self.msg)
        except KeyboardInterrupt:
            print("Publish error!")
        end_time = time.time()
        execute_time = end_time - start_time
        print("Excute time: ", int(execute_time*1000), "ms")
        # Display the output
        cv2.imshow("Image depth", cv_image)
        #cv2.imshow("Image depth", depth_img)
        cv2.waitKey(3)
        cv2.imshow("Image quality", dis_img)
        cv2.waitKey(3)

def main(args):
    rospy.init_node('grasp_prediction', anonymous=True)
    grasp_class = grasp_prediction()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down grasp prediction!")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
