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
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge, CvBridgeError
# Auxiliary function
from ggcnn_pred_keras import prediction, process_output, process_output2
from ggcnn_pred_keras import processing_depth, filter_thresh, local_max
from ggcnn.msg import grasp_point
from models.ggcnn_keras import ggcnn
from coordinate import coordinate_extract
import tensorflow as tf

cam_param =  {'cam_fx': 606,
              'cam_fy': 605,
              'cam_tx': 245,
              'cam_ty': 316}

class grasp_prediction:
    def __init__(self):
        self.bridge = CvBridge()
        self.msg = grasp_point()
        self.grasp_pub = rospy.Publisher("grasp_config", grasp_point, queue_size=3)
        self.pcd_pub = rospy.Publisher("/grasp_pointcloud", PointCloud2, queue_size=3)
        self.depth_node = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",
                                            Image, self.depth_callback, queue_size=1)
        self.color_node = rospy.Subscriber("/camera/color/image_raw",
                                            Image, self.color_callback, queue_size=1)
        self.depth_img = np.zeros((300, 300))
        self.coordinate = coordinate_extract(cam_param)

    def depth_callback(self, depth_msg):
        depth_image = CvBridge().imgmsg_to_cv2(depth_msg)
        self.depth_img = depth_image

    def color_callback(self, color_msg):
        start_time = time.time()
        # Invoke the image from camera
        ori_image = CvBridge().imgmsg_to_cv2(color_msg)
        use_image = ori_image.copy()
        # Process the image
        rgb_img = processing_depth(use_image)
        # Predict the grasp
        q_img, cos_img, sin_img, width_img = prediction(rgb_img)
        # Filter the q_img
        ret = filter_thresh(q_img[0], 0.5)
        q_img_process = local_max(q_img[0])
        # Post process the output
        dis_img, grasp_pts, center = process_output(rgb_img[0], q_img[0], cos_img[0], sin_img[0], width_img[0])

        offsets = np.array([170, 90])
        dis_img2, grasp_pts_list, centers, angles = process_output2(rgb_img[0], q_img[0], cos_img[0], sin_img[0], width_img[0], offsets)
        # Publish the grasp points
        depth_point = self.depth_img[center[0]+90, center[1]+170]
        px, py, pz = self.coordinate.get_coord(depth_point, center[0]+90, center[1]+170)
        pxs, pys, pzs = self.coordinate.get_coords(self.depth_img, centers[:,0]+90, centers[:,1]+170)
        # print(pxs, pys, pzs)
        #print("Depth point: ", depth_point)
        grasp_pts = grasp_pts + offsets

        # Find the coordinate for grasping
        grasp_pcd, pcl2_pcd = self.coordinate.get_pcd(self.depth_img, grasp_pts, grasp_pts_list)

        self.msg.px = px/1000.0
        self.msg.py = py/1000.0
        self.msg.pz = pz/1000.0

        pxs = list(np.array(pxs)/1000.0)
        pys = list(np.array(pys)/1000.0)
        pzs = list(np.array(pzs)/1000.0)
        costhetas = list(np.array(angles)[:, 0])
        sinthetas = list(np.array(angles)[:, 1])
        self.msg.pxs = pxs
        self.msg.pys = pys
        self.msg.pzs = pzs
        self.msg.costhetas = costhetas
        self.msg.sinthetas = sinthetas

        # Convert from depth_img to JET
        cv2.polylines(self.depth_img, [grasp_pts], True, (0, 255, 0))

        # Publish the grasp points and pointcloud
        try:
            self.pcd_pub.publish(pcl2_pcd)
            self.grasp_pub.publish(self.msg)
        except KeyboardInterrupt:
            print("Publish error!")
        end_time = time.time()
        execute_time = end_time - start_time
        print("Excute time: ", int(execute_time*1000), "ms")
        # Display the output
        #heatmap = cv2.applyColorMap(np.uint8(q_img[0]*255), cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(np.uint8(ret*255), cv2.COLORMAP_JET)
        q_img_process = cv2.applyColorMap(np.uint8(q_img_process*255), cv2.COLORMAP_JET)
        depth_dis = np.uint8(np.clip(self.depth_img, 0, 1000)/1000.0*255)
        cv2.imshow("Depth", depth_dis)
        cv2.waitKey(3)
        cv2.imshow("Multi-grasp", dis_img2)
        cv2.waitKey(3)
        cv2.imshow("Grasp Quality", heatmap)
        cv2.waitKey(3)
        cv2.imshow("Best Grasp", dis_img)
        cv2.waitKey(3)

def main(args):
    rospy.init_node('grasp_prediction', anonymous=True)
    grasp_class = grasp_prediction()
    while not rospy.is_shutdown():
        rospy.spin()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
