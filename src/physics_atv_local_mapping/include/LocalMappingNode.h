#include <pcl/point_types.h>
#include <pcl/features/feature.h>
#include <pcl/conversions.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h> 
#include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

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

#include <boost/circular_buffer.hpp>

#include "ScrollGrid.h"

typedef pcl::PointXYZRGB  PointT;

using namespace sensor_msgs;
using namespace message_filters;
using namespace pcl;
using namespace geometry_msgs;
using namespace std;
using namespace Eigen;

class LocalMappingNode
{
public:
    LocalMappingNode();
    ~LocalMappingNode();

protected:
    ros::NodeHandle nh_;

    std::string robot, pointcloud_topic, pose_topic, odom_topic, platform;
    double resolution, max_x, min_x, max_y, min_y;
    int map_height, map_width;
    bool first_msg_received;
    double downsample_leafsize_before, downsample_leafsize_after, remove_outlier_thresh;
    bool downsample_pointcloud_before, downsample_pointcloud_after, filter_outliers; 
    int remove_outlier_count, downsample_pointcloud_after_thresh, max_points_in_cloud;
    bool inflate_fill_hole;
    bool publish_pc_for_debug, visualize_maps, align_gravity;

    int pc_frame_skip;
    int pc_buffer_size;
    int pc_counter;
    boost::circular_buffer<pcl::PointCloud<PointT>> pc_buffer; ///< buffer contaning the most recent pointclouds
    boost::circular_buffer<geometry_msgs::Pose> pc_tf_buffer; ///< buffer contaning the most recent pointclouds

    ScrollGrid localmap;
    ros::Publisher height_publisher, rgb_publisher;
    ros::Publisher height_inflate_publisher, rgb_inflate_publisher;
    pcl::PointCloud<PointT> pointcloud2_current, pointcloud2_merged, pointcloud2_merged_ground;
    Eigen::Matrix4f t_ground2cam, t_body2novatel;
    Eigen::Quaternionf latest_ori; // for gravity alignment
    pcl::VoxelGrid<PointT> vg; // for downsampling of pointclouds
    pcl::StatisticalOutlierRemoval<PointT> sor;

    // ros::Subscriber pointcloud_subscriber_;
    // ros::Subscriber transform_subscriber_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sync;
    // message_filters::Subscriber<geometry_msgs::TransformStamped> trans_sync;
    message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sync;
    ros::Subscriber odom_subscriber; // use the orientation to compensate the angle, for better slope reconstruction

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, geometry_msgs::PoseStamped> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    geometry_msgs::Pose last_pose, current_pose;

    void process_data(const sensor_msgs::PointCloud2ConstPtr &pc_msg, 
                    const geometry_msgs::PoseStampedConstPtr &pose_msg);
    void downsamplePointCloud(pcl::PointCloud<PointT> &pointcloud, double leafsize);
    pcl::PointCloud<PointT> convertFromMsgToPointCloud(const sensor_msgs::PointCloud2& pointcloud_msg);
    Eigen::Matrix4f transform2Matrix(const geometry_msgs::TransformStamped trans_msg);
    // void handle_pc(const sensor_msgs::PointCloud2& pc_msg);
    // void handle_trans(const geometry_msgs::TransformStamped& trans_msg);
    void transformPointCloud(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, 
                                const geometry_msgs::Pose pose_init, const geometry_msgs::Pose pose_target);
    void alignGravity(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, Eigen::Quaternionf quat);
    Eigen::Matrix4f pose2Matrix(geometry_msgs::Pose pose);
    void filterPointCloudByRange(pcl::PointCloud<PointT> &input_pc);
    pcl::PointCloud<PointT> filterPointCloudByNumber(pcl::PointCloud<PointT> &input_pc);
    void publishImages(uint8_t* colormap, float* heightmap,
                        ros::Publisher& rgb_pub, ros::Publisher& color_pub, 
                        ros::Time stamp, std::string frame_id);
    void handle_odom(const nav_msgs::Odometry& odom_msg);

    // for debug
    ros::Publisher pc_publisher;
    void publishPointCloud(const pcl::PointCloud<PointT> &pointcloud, ros::Time stamp, std::string frame_id_);
    // void test_eigen_transform();


};
