import numpy as np
import pcl
import struct
from skimage.draw import polygon
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointField
import std_msgs.msg
import rospy

class coordinate_extract():
    def __init__(self, cam_param):
        self.cam_fx = cam_param['cam_fx']
        self.cam_fy = cam_param['cam_fy']
        self.cam_tx = cam_param['cam_tx']
        self.cam_ty = cam_param['cam_ty']


    def get_pcd(self, depth_img, grasp_pts, grasp_pts_list = None):
        "Get points cloud from depth and grasp_pts"
        cc = None
        # Entire pcd of image
        points_entire_img = self.get_points(depth_img)
        # Get polygon points
        for _, elem in enumerate(grasp_pts_list):
            cci, rri = polygon(elem[:,0], elem[:,1])
            if cc is None:
                cc = cci
                rr = rri
            else:
                cc = np.append(cc, cci)
                rr = np.append(rr, rri)
        # print(cc.shape)
        # cc, rr = polygon(grasp_pts[:,0], grasp_pts[:,1])
        # Convert to points
        num_points = len(rr)
        points = np.zeros((num_points, 4), np.float32)
        for i in range(0, num_points):
            depth = depth_img[rr[i], cc[i]]
            points[i] = self.get_point(depth, rr[i], cc[i])
            points_entire_img[rr[i]*640 + cc[i]] = points[i]
        # To pcl2 format
        header, fields = self.pcl_head_field()
        #pcl2_pcd = pcl2.create_cloud_xyz32(header, points)
        #grasp_pcd = pcl.PointCloud(np.array(points, dtype=np.float32))
        #grasp_pcd = pcl.PointCloud_PointXYZRGBA(points)
        #grasp_pcd = pcl.PointCloud_PointXYZRGB(points)
        #grasp_pcd = 0
        pcl2_pcd = pcl2.create_cloud(header, fields, points)
        pcl2_pcd = pcl2.create_cloud(header, fields, points_entire_img)
        grasp_pcd = 0
        return grasp_pcd, pcl2_pcd

    def get_point(self, depth, u, v, r=int(0), g=int(255), b=int(255), a=255):
        pz = depth
        py = (u-self.cam_tx)*1.0*pz/self.cam_fx
        px = (v-self.cam_ty)*1.0*pz/self.cam_fy
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        point = np.array([px/1000.0, py/1000.0, pz/1000.0, rgb])
        point = point.astype(np.float32)
        return point

    def get_coord(self, depth, u, v):
        pz = int(depth)
        py = int((u-self.cam_tx)*1.0*pz/self.cam_fx)
        px = int((v-self.cam_ty)*1.0*pz/self.cam_fy)
        return px, py, pz

    def get_coords(self, depth_img, us, vs):
        pxs = []
        pys = []
        pzs = []
        # print(type(us))
        for i in range(us.shape[0]):
            pz = int(depth_img[us[i], vs[i]])
            py = int((us[i]-self.cam_tx)*1.0*pz/self.cam_fx)
            px = int((vs[i]-self.cam_ty)*1.0*pz/self.cam_fy)
            pxs.append(px)
            pys.append(py)
            pzs.append(pz)
        return pxs, pys, pzs

    def get_points(self, depth_img):
        # Mask
        cols = []
        rows = []
        height = depth_img.shape[0]
        width = depth_img.shape[1]
        for i in range(0, height):
            row = i*np.ones([width, 1])
            rows.append(row)
        for i in range(0, width):
            col = i*np.ones([height, 1])
            cols.append(col)
        rows = np.array(rows)[:,:, 0]
        cols = np.array(cols)[:,:, 0]
        cols = np.transpose(cols)
        # Mask of color
        r = int(255)
        g = int(255)
        b = int(255)
        a = 0
        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        color = np.ones((height*width, 1), np.float32)
        color = color*rgb
        # Points
        pzs = depth_img
        pys = (rows-self.cam_tx)*1.0*pzs/self.cam_fx
        pxs = (cols-self.cam_ty)*1.0*pzs/self.cam_fy
        pxs = pxs.flatten(order='C')
        pys = pys.flatten(order='C')
        pzs = pzs.flatten(order='C')
        points = np.concatenate([pxs/1000.0, pys/1000.0, pzs/1000.0, color[:,0]])
        points = points.reshape((4, height*width)).transpose()
        return points

    def pcl_head_field(self):
        #header
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.FLOAT32, 1),
                  ]

        return header, fields

    def surface_normal_pcd(grasp_pcd, center):
        center_p = get_point(center)[0:3]
        # Find normal surface at center

        x, y, z = 0, 0, 0
        alpha, beta, gamma = 0, 0, 0
        return x, y, z, alpha, beta, gamma
