#!/usr/bin/env python

import argparse
import logging
from os import path
import sys
import numpy as np
import cv2
#import torch.utils.data
import torch
from skimage.io import imread
import matplotlib.pyplot as plt

#logging.basicConfig(level=logging.INFO)
color_img = imread("pcd0100r.png")
# Load the model
MODEL_FILE = "models/epoch_20_iou_0.00"
here = path.dirname(path.abspath(__file__))
sys.path.append(here)
path = path.join(path.dirname(__file__), MODEL_FILE)
#print(path)
model = torch.load(path)
#model = torch.load(MODEL_FILE)
device = torch.device("cuda:0")


# Crop and resize the image
def processing_depth(depth, crop_size=300, out_size=300):
    imgheight, imgwidth = depth.shape
    depth_crop = depth[(imgheight-crop_size)/2:(imgheight+crop_size)/2,
                        (imgwidth-crop_size)/2:(imgwidth+crop_size)/2]

    # Scale the depth into 0-1
    #depth_scale = depth_crop-np.mean(depth_crop)
    #depth_scale = np.clip(depth_scale, -1, 1)
    depth_scale = 2.0*(depth_crop-np.min(depth_crop))/(np.max(depth_crop)-np.min(depth_crop))-1
    #print(np.min(depth_crop))
    #print(depth_scale)

    # Resize to the out_size
    depth_resize = cv2.resize(depth_scale, (out_size, out_size), cv2.INTER_NEAREST)

    return depth_resize
"""
# Crop and resize the image
def processing_depth(depth, crop_size=300, out_size=300):
    imgheight, imgwidth = depth.shape
    depth_crop = depth[(imgheight-crop_size)/2:(imgheight+crop_size)/2,
                        (imgwidth-crop_size)/2:(imgwidth+crop_size)/2]

    # Scale the depth into 0-1
    depth_scale = depth_crop-np.mean(depth_crop)
    depth_scale = np.clip(depth_scale, -1, 1)

    # Resize to the out_size
    depth_resize = cv2.resize(depth_scale, (out_size, out_size), cv2.INTER_NEAREST)

    return depth_resize
"""

def process_output(display_img, q_img, cos_img, sin_img, width_img, grip_length=30):
    #1. Find the max quality pixel
    result = np.where(q_img == np.amax(q_img))
    # zip the 2 arrays to get the exact coordinates
    listOfCordinates = list(zip(result[0], result[1]))
    (x, y) = listOfCordinates[0]
    center = np.array([x, y])
    #2. Calculate the angle
    grasp_cos = cos_img[x, y]
    grasp_sin = sin_img[x, y]

    width = width_img[x, y]*150
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
    #4. Draw in the images
    color = (0, 255, 0)
    cv2.polylines(display_img, [pts], True, color )
    return display_img, pts, center

def predict(depth, crop_size=300, output_size=300):
    depth_process = processing_depth(depth)
    depthT = torch.from_numpy(depth_process.reshape(1, 1, output_size, output_size).astype(np.float32)).to(device)

    with torch.no_grad():
        pred_out = model(depthT)

    # Get the quality image
    q_img = pred_out[0].cpu().numpy().squeeze()
    cos_img = pred_out[1].cpu().numpy().squeeze()
    sin_img = pred_out[2].cpu().numpy().squeeze()
    width_img = pred_out[3].cpu().numpy().squeeze()

    # Process the output
    color_crop = color_img[int((color_img.shape[0] - 300)/2):int((color_img.shape[0] + 300)/2),
                                int((color_img.shape[1] - 300)/2):int((color_img.shape[1] + 300)/2)]
    output_img = process_output(color_crop, q_img, cos_img, sin_img, width_img)
    """
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(depth_process, cmap='gray')
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 2)
    plot = ax.imshow(output_img)
    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(1, 3, 3)
    plot = ax.imshow(q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    print(q_img.shape)
    print(np.max(q_img))
    plt.show()
    """

    return q_img, cos_img, sin_img, width_img
