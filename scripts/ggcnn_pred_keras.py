#!/usr/bin/env python

import argparse
import logging
from os import path
import sys
import numpy as np
import cv2
import tensorflow as tf
#import torch.utils.data
from models.ggcnn_keras import ggcnn
from models.ggcnn2 import ggcnn2
from skimage.io import imread
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as k

session = tf.Session(graph=tf.Graph())
global session
with session.graph.as_default():
    k.set_session(session)
    print("Creating model ...")
    model = ggcnn2().model()
    print("Creating model done!!!")
    print("Loading model...")
    model.load_weights('models/ggcnn_weight_2.h5')
    model._make_predict_function()
    print("Load model done!!!")
    print("Load model done!!!")
    print("Load model done!!!")
    print("Load model done!!!")
    print("Load model done!!!")
    print("Load model done!!!")

# Crop and resize the image
def processing_depth(rgb, crop_size=300, out_size=300):
    imgheight, imgwidth = rgb.shape[0], rgb.shape[1]
    # Crop
    rgb = rgb[int((imgheight-crop_size)/2):int((imgheight+crop_size)/2),
                        int((imgwidth-crop_size)/2):int((imgwidth+crop_size)/2)]

    # Scale
    rgb_scale = (rgb-128.0)/255.0
    # Resize to the out_size
    rgb_resize = cv2.resize(rgb_scale, (out_size, out_size), cv2.INTER_NEAREST)
    rgb_resize = np.expand_dims(rgb_resize, axis=0)
    return rgb_resize

def process_output(display_img, q_img, cos_img, sin_img, width_img, grip_length=30, normalize=True):
    #0. Convert image if normalize:
    if normalize:
        display_img = np.uint8(display_img*255+128)
    #1. Find the max quality pixel
    result = np.where(q_img == np.amax(q_img))
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(result[0], result[1]))
    (x, y) = listOfCordinates[0]
    center = np.array([x, y])
    #2. Calculate the angle
    grasp_cos = cos_img[x, y]
    grasp_sin = sin_img[x, y]
    # Normalize sine&cosine of 2*phi
    sum_sqrt = np.sqrt(grasp_cos**2+grasp_sin**2)
    grasp_cos = grasp_cos/sum_sqrt
    grasp_sin = grasp_sin/sum_sqrt
    # Because of symmetric so we can use this
    grasp_sin_modi = np.sqrt((1-grasp_cos)/2)
    grasp_cos_modi = grasp_sin/(2*grasp_sin_modi)
    grasp_cos = grasp_cos_modi
    grasp_sin = grasp_sin_modi

    #width = width_img[x, y]*150
    width = 60
    #3. Find the grasp BoundingBoxes
    x1 = width/2.0
    y1 = grip_length/2.0
    x2 = width/2.0
    y2 = -grip_length/2.0
    x3 = -width/2.0
    y3 = grip_length/2.0
    x4 = -width/2.0
    y4 = -grip_length/2.0
    # Rotate the angle
    R = np.array([[grasp_cos, grasp_sin], [-grasp_sin, grasp_cos]])
    pt1 = np.matmul(R, np.array([x1, y1]))
    pt2 = np.matmul(R, np.array([x2, y2]))
    pt3 = np.matmul(R, np.array([x3, y3]))
    pt4 = np.matmul(R, np.array([x4, y4]))

    pt1[0] = y + pt1[0]
    pt1[1] = x + pt1[1]
    pt2[0] = y + pt2[0]
    pt2[1] = x + pt2[1]
    pt3[0] = y + pt3[0]
    pt3[1] = x + pt3[1]
    pt4[0] = y + pt4[0]
    pt4[1] = x + pt4[1]

    pts = np.array([pt1, pt3, pt4, pt2], np.int32)
    #print(R)
    #4. Draw in the images
    color = (0, 255, 255)
    cv2.polylines(display_img, [pts], True, color )
    return display_img, pts, center

def prediction(rgb, crop_size=300, output_size=300):
    with session.graph.as_default():
        k.set_session(session)
        pred = model.predict(rgb)
    # Get the quality image
    q_img = pred[...,0]
    width_img = pred[...,1]
    cos_img = pred[...,2]
    sin_img = pred[...,3]
    color_crop = rgb[0]
    return q_img, cos_img, sin_img, width_img

# No need to use ^^
def filter_thresh(q_img, threshold=0.5):
    ret, thresh1 = cv2.threshold(q_img, 0.5, 1, cv2.THRESH_TOZERO)
    return thresh1

def local_max(q_img, threshold=0.5):
    height, width = q_img.shape[0], q_img.shape[1]
    kernel = np.ones((25,25),np.uint8)
    dilation = cv2.dilate(q_img,kernel,iterations = 1)
    fil_nonzero = q_img - dilation
    local_max = []
    for i in range(0, height):
        for j in range(0, width):
            if dilation[i,j] == q_img[i,j] and q_img[i,j] > threshold:
                local_max.append((i, j))
    return local_max

def draw_output(img, pts_list, color=(255, 0, 0)):
    for _, elem in enumerate(pts_list):
        cv2.polylines(img, [elem], True, color )
    return img

def gripper_cal(center, width, cos, sin, grip_length=30):
    x, y = center
    #1. Calculate the angle
    grasp_cos = cos
    grasp_sin = sin
    # Normalize sine&cosine of 2*phi
    sum_sqrt = np.sqrt(grasp_cos**2+grasp_sin**2)
    grasp_cos = grasp_cos/sum_sqrt
    grasp_sin = grasp_sin/sum_sqrt
    # Because of symmetric so we can use this
    grasp_sin_modi = np.sqrt((1-grasp_cos)/2)
    grasp_cos_modi = grasp_sin/(2*grasp_sin_modi)
    grasp_cos = grasp_cos_modi
    grasp_sin = grasp_sin_modi

    #width = width_img[x, y]*150
    width = 60
    #2. Find the grasp BoundingBoxes
    x1 = width/2.0
    y1 = grip_length/2.0
    x2 = width/2.0
    y2 = -grip_length/2.0
    x3 = -width/2.0
    y3 = grip_length/2.0
    x4 = -width/2.0
    y4 = -grip_length/2.0
    # Rotate the angle
    R = np.array([[grasp_cos, grasp_sin], [-grasp_sin, grasp_cos]])
    pt1 = np.matmul(R, np.array([x1, y1]))
    pt2 = np.matmul(R, np.array([x2, y2]))
    pt3 = np.matmul(R, np.array([x3, y3]))
    pt4 = np.matmul(R, np.array([x4, y4]))

    pt1[0] = y + pt1[0]
    pt1[1] = x + pt1[1]
    pt2[0] = y + pt2[0]
    pt2[1] = x + pt2[1]
    pt3[0] = y + pt3[0]
    pt3[1] = x + pt3[1]
    pt4[0] = y + pt4[0]
    pt4[1] = x + pt4[1]

    pts = np.array([pt1, pt3, pt4, pt2], np.int32)
    angle = np.array([grasp_cos, grasp_sin])
    return pts, angle

def process_output2(display_img, q_img, cos_img, sin_img, width_img, offsets, normalize=True):
    if normalize:
        display_img = np.uint8(display_img*255+128)
    local_max_points = local_max(q_img)
    pts_list = []
    pts_list_nooffset = []
    angles = []
    for i in range(len(local_max_points)):
        coord = local_max_points[i]
        pts, angle = gripper_cal(coord, width_img[coord], cos_img[coord], sin_img[coord])
        pts_list_nooffset.append(pts)
        pts = pts + offsets
        pts_list.append(pts)
        angles.append(angle)
    output_dis = draw_output(display_img, pts_list_nooffset)
    centers = np.array(local_max_points)
    return output_dis, pts_list, centers, angles
