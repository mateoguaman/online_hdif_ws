#include <pcl/point_types.h>
#include <pcl/features/feature.h>
#include <pcl/conversions.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h> 
#include <message_filters/subscriber.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include "pcl/filters/statistical_outlier_removal.h" // to filter outliers
#include "pcl/filters/voxel_grid.h" //for downsampling the point cloud
#include <pcl/filters/passthrough.h>
#include "pcl/registration/transforms.h" //for the transformation function

#include <ctime>
#include <chrono>
#include <cmath>

#include "ScrollGrid.h"


typedef pcl::PointXYZRGB  PointT;

using namespace sensor_msgs;
using namespace message_filters;
using namespace pcl;
using namespace geometry_msgs;
using namespace std;
using namespace Eigen;

class PointProjectionNode
{
public:
    PointProjectionNode();
    ~PointProjectionNode();

private:
    ros::NodeHandle nh_;

    std::string pointcloud_topic, pose_topic, platform;
    double resolution, max_x, min_x, max_y, min_y;
    int map_height, map_width;
    bool inflate_fill_hole;
    bool publish_pc_for_debug, visualize_maps, align_gravity;

    ScrollGrid localmap;
    ros::Publisher height_publisher, rgb_publisher;
    ros::Publisher height_inflate_publisher, rgb_inflate_publisher;
    pcl::PointCloud<PointT> pointcloud2_current, pointcloud2_current_ground;
    Eigen::Matrix4f t_ground2cam, t_body2novatel;
    Eigen::Quaternionf latest_ori; // for gravity alignment

    ros::Subscriber pc_subscriber;
    ros::Subscriber odom_subscriber; // use the orientation to compensate the angle, for better slope reconstruction

    pcl::PointCloud<PointT> convertFromMsgToPointCloud(const sensor_msgs::PointCloud2& pointcloud_msg);
    Eigen::Matrix4f transform2Matrix(const geometry_msgs::TransformStamped trans_msg);
    void handle_pc(const sensor_msgs::PointCloud2ConstPtr& pc_msg);
    // void handle_trans(const geometry_msgs::TransformStamped& trans_msg);
    void alignGravity(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, Eigen::Quaternionf quat);
    void publishImages(uint8_t* colormap, float* heightmap,
                        ros::Publisher& rgb_pub, ros::Publisher& color_pub, 
                        ros::Time stamp, std::string frame_id);
    void handle_odom(const nav_msgs::Odometry& odom_msg);

    // for debug
    ros::Publisher pc_publisher;
    void publishPointCloud(const pcl::PointCloud<PointT> &pointcloud, ros::Time stamp, std::string frame_id_);
    // void test_eigen_transform();


};