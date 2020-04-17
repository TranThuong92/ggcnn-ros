/*
Date:     2020/04/09
Composer: Tran Ngoc Cong Thuong
This code is for ggcnn network to custom display point cloud
 */
 #include "pcl_online_viewer.h"
 #include <pcl/visualization/cloud_viewer.h>
 #include <pcl/visualization/pcl_visualizer.h>

#if (defined PCL_MINOR_VERSION && (PCL_MINOR_VERSION >= 7))
#include <pcl_conversions/pcl_conversions.h>
typedef pcl::PCLPointCloud2 PointCloud2;
#else
typedef sensor_msgs::PointCloud2 PointCloud2;
#endif

using namespace std;
ostringstream strg;

typedef pcl::PointXYZ Point;
typedef PointCloud2::ConstPtr PointCloudConstPtr;

boost::mutex m;

// Grasp points from grasp configuration
float px=0;
float py=0;
float pz=0.5;
float pxs[100];
float pys[100];
float pzs[100];
float costhetas[100];
float sinthetas[100];
float costheta = 1;
float sintheta = 0;
int center_size = 0;
int center_size_old = 0;

// Normal estimation
int num_neighbor=200;
std::vector<int> pointIdxNKNSearch(num_neighbor);
std::vector<float> pointNKNSquareDistance(num_neighbor);
pcl::Normal normal;
Eigen::Vector4f plane_parameters;
Eigen::Affine3f transform_coord = Eigen::Affine3f::Identity();
Eigen::Matrix3f rotation_img;
Eigen::Matrix3f rotation;
pcl::PointXYZRGBA searchPoint;
float curvature;
bool ne_bool;

// Get point cloud variales
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;

// Display variables
int display_flag = 0;
int center_cb_flag = 0;
int cloud_cb_flag = 0;
int ne_done = 0;

// Function of kd-tree
void knearest(int i)
{
  searchPoint.x = pxs[i];
  searchPoint.y = pys[i];
  searchPoint.z = pzs[i];

  costheta = costhetas[i];
  sintheta = sinthetas[i];

  ROS_INFO("Grasp point is %f %f %f", searchPoint.x, searchPoint.y, searchPoint.z);

  kdtree.nearestKSearch(searchPoint, num_neighbor, pointIdxNKNSearch, pointNKNSquareDistance);
}

// Find point normal estimation
void pointnormalestimation()
{
    //Convert point cloud
    computePointNormal(*temp_cloud, pointIdxNKNSearch, plane_parameters, normal.curvature);
}

// Call back function for center subscribe
void center_cb(const ggcnn::grasp_point::ConstPtr &msg)
{
  px = msg->px;
  py = msg->py;
  pz = msg->pz;
  center_size = msg->pxs.size();
  for (int i=0; i<center_size; i++)
  {
    pxs[i] = msg->pxs[i];
    pys[i] = msg->pys[i];
    pzs[i] = msg->pzs[i];
    costhetas[i] = msg->costhetas[i];
    sinthetas[i] = msg->sinthetas[i];
    std::cout << pxs[i] << "\n";
  }
  center_cb_flag = 1;
}

//Translate and rotate coordinate for displaying
void virtual_coord()
{
  //Restart coordinate
  transform_coord = Eigen::Affine3f::Identity();

  float theta = M_PI/4; //Need to read from grasp_coordinate
  ROS_INFO("Grasp angle %f %f ", costheta, sintheta);
  // Rotation from image
  rotation_img (0, 0) = sintheta;
  rotation_img (0, 1) = -costheta;
  rotation_img (0, 2) = 0;
  rotation_img (1, 0) = costheta;
  rotation_img (1, 1) = sintheta;
  rotation_img (1, 2) = 0;
  rotation_img (2, 0) = 0;
  rotation_img (2, 1) = 0;
  rotation_img (2, 2) = 1;

  // z axis
  rotation (0, 2) = plane_parameters(0);
  rotation (1, 2) = plane_parameters(1);
  rotation (2, 2) = plane_parameters(2);
  // y axis
  rotation (0, 1) = plane_parameters(2)/sqrt(plane_parameters(0)*plane_parameters(0)
                                            +plane_parameters(2)*plane_parameters(2));
  rotation (1, 1) = 0;
  rotation (2, 1) = plane_parameters(0)/sqrt(plane_parameters(0)*plane_parameters(0)
                                            +plane_parameters(2)*plane_parameters(2));
  // x axis
  rotation (0, 0) = rotation(1,1)*rotation(2,2) - rotation(2,1)*rotation(1,2);
  rotation (1, 0) = rotation(2,1)*rotation(0,2) - rotation(0,1)*rotation(2,2);
  rotation (2, 0) = rotation(0,1)*rotation(1,2) - rotation(1,1)*rotation(0,2);

  //Roatation
  transform_coord.rotate(rotation);
  transform_coord.rotate(rotation_img);

  transform_coord.translation() << searchPoint.x, searchPoint.y, searchPoint.z;
}

// Callback of cloud
void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
     // ROS_INFO("inside callback");
     pcl::PCLPointCloud2 pcl_pc2;
     pcl_conversions::toPCL(*cloud_msg, pcl_pc2);

     pcl::fromPCLPointCloud2(pcl_pc2,*temp_cloud);

     m.lock();
     ne_done = 1;
     kdtree.setInputCloud(temp_cloud);

     m.unlock();
     cloud_cb_flag = 1;
}

int main (int argc, char** argv)
{
     ros::init (argc, argv, "cloud_sub");
     ros::NodeHandle nh;
     ros::Rate loop_rate(10);

     // Display PCL
     pcl::visualization::PCLVisualizer viewer("PCL Viewer");
     viewer.setBackgroundColor (0.0, 0.0, 0.5);

     ros::Subscriber sub;
     sub = nh.subscribe ("input", 1, cloud_callback);
     ros::Subscriber sub_center = nh.subscribe ("grasp_config", 1, center_cb);

     while (nh.ok ())
     {
        ros::spinOnce();
        ros::Duration (0.001).sleep ();

        viewer.spinOnce (10);

        //viewer.removeAllCoordinateSystems();
        // m.lock();
        // reset the viewer first
        //viewer.removePointCloud();
        for (int idx=0; idx<center_size_old; idx++)
        {
            //viewer.removeAllCoordinateSystems();
        }

        //add new point cloud to the viewer
        if (ne_done)
        {
          if(center_cb_flag == 1 && cloud_cb_flag == 1)
          {
            ROS_INFO("HERE!");
            // Remove the old point cloud
            viewer.removePointCloud();
            for (int idx=0; idx<center_size_old; idx++)
            {
                strg << idx;
                viewer.removeCoordinateSystem(strg.str(), 0);
            }
            viewer.removeAllCoordinateSystems();
            // Refresh point cloud
            viewer.addPointCloud<pcl::PointXYZRGBA>(temp_cloud);
            kdtree.setInputCloud(temp_cloud);
            center_size_old = center_size;
            for(int idx=0; idx<center_size; idx++)
            {
              center_size_old = center_size;
              knearest(idx);
              pointnormalestimation();
              virtual_coord();
              strg << idx;
              viewer.addCoordinateSystem (0.05, transform_coord, strg.str(), 0);
            }
            center_cb_flag = 0;
            cloud_cb_flag = 0;
          }
        }
        // m.unlock();
     }
}
