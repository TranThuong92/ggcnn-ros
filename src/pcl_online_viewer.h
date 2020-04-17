#include <ros/ros.h>
#include <pcl/io/io.h>
#include <sensor_msgs/PointCloud2.h>
#include <ggcnn/grasp_point.h>
#include <boost/thread/mutex.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <vector>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>       /* sqrt */
#include <string>

void knearest();
void pointnormalestimation();
void center_cb(const ggcnn::grasp_point::ConstPtr &msg);
void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
