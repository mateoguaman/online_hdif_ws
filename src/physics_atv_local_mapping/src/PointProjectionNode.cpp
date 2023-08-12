/*
Project single point could on to the grid map
*/
#include "PointProjectionNode.h"

PointProjectionNode::PointProjectionNode(): nh_("~")
{
    ROS_INFO ("Local Mapping node initializing.");
    nh_.param<std::string>("pointcloud_topic", pointcloud_topic, "/deep_cloud");
    nh_.param("resolution", resolution, 0.02);
    nh_.param("max_x", max_x, 10.0);
    nh_.param("min_x", min_x, 0.0);
    nh_.param("max_y", max_y, 5.0);
    nh_.param("min_y", min_y, -5.0);
    nh_.param("inflate_fill_hole", inflate_fill_hole, true);
    nh_.param("visualize_maps", visualize_maps, false); // downsample the pc only when points number is big 
    nh_.param("publish_pc_for_debug", publish_pc_for_debug, true); // downsample the pc only when points number is big 
    nh_.param("align_gravity", align_gravity, true); // align the point cloud with gravity according to the odom message 
    nh_.param<std::string>("platform", platform, "yamaha"); // yamaha, racer, warthog

    localmap.init(resolution, min_x, max_x, min_y, max_y); // = ScrollGrid(resolution, min_x, max_x, min_y, max_y); 
    map_height = localmap.getMapHeight();
    map_width = localmap.getMapWidth();

    height_publisher = nh_.advertise<Image> ("/local_height_map", 100);
    rgb_publisher = nh_.advertise<Image>("/local_rgb_map", 100);
    if(inflate_fill_hole)
    {
        height_inflate_publisher = nh_.advertise<Image> ("/local_height_map_inflate", 100);
        rgb_inflate_publisher = nh_.advertise<Image>("/local_rgb_map_inflate", 100);        
    }

    pc_subscriber = nh_.subscribe(pointcloud_topic, 1, &PointProjectionNode::handle_pc, this);
    odom_subscriber = nh_.subscribe("/odometry/filtered_odom", 100, &PointProjectionNode::handle_odom, this);


    // the transform of the multisense camera in the projected ground frame
    if(platform == "yamaha")
    {
        t_ground2cam << (Eigen::Matrix4f() << 0.9692535,   0.00610748, -0.24598853,   0.0, 
                                            0.,         -0.99969192, -0.02482067, 0.0,
                                            -0.24606434,   0.02405752,  -0.96895489,   1.77348523,
                                            0,   0,  0,           1.0).finished();
    }
    else if(platform == "racer")
    {
        t_ground2cam << (Eigen::Matrix4f() << 0.90630779,  0.        , -0.42261826,   0.0, 
                                            -0.        , -1.        , -0.        , 0.0,
                                            -0.42261826,  0.        , -0.90630779,   1.77348523,
                                            0,   0,  0,           1.0).finished();
    }
    else if(platform == "warthog")
    {
        t_ground2cam << (Eigen::Matrix4f() << 1.0,  0.  , 0.0,   0.0, 
                                            -0.  , -1. , -0.        , 0.0,
                                            -0.0,  0.  , -1.0,   0.77348523,
                                            0,   0,  0,   1.0).finished();
    }
                                        
    t_body2novatel << (Eigen::Matrix4f() << 0.0, 1.0, 0.0, 0.0, 
                                            -1., 0.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0,
                                            0.0, 0.0, 0.0, 1.0).finished();
    // TODO: the synchronization of odom and pc/pose could be an issue
    latest_ori = Eigen::Quaternionf(1.0, 0.0, 0.0, 0.0);

    // for debug
    if(publish_pc_for_debug)
        pc_publisher = nh_.advertise<PointCloud2>("merged_pointcloud", 100);

    ROS_INFO ("Local Mapping node initialized.");
}

PointProjectionNode::~PointProjectionNode()
{

}

pcl::PointCloud<PointT> PointProjectionNode::convertFromMsgToPointCloud(const sensor_msgs::PointCloud2& pointcloud_msg)
{
    // auto start = std::chrono::steady_clock::now();
    pcl::PointCloud<PointT> pointcloud_pcl; //, pointcloud_downsampled_pcl;
    pcl::fromROSMsg(pointcloud_msg, pointcloud_pcl);
    return (pointcloud_pcl);
}

void PointProjectionNode::publishPointCloud(const pcl::PointCloud<PointT> &pointcloud, ros::Time stamp, string frame_id_)
{
    // auto start = std::chrono::steady_clock::now();
    pcl::PCLPointCloud2 pointcloud2;
    pcl::toPCLPointCloud2(pointcloud, pointcloud2);

    sensor_msgs::PointCloud2 my_cloud;
    pcl_conversions::fromPCL(pointcloud2, my_cloud);

    my_cloud.header.frame_id = frame_id_;
    my_cloud.header.stamp = stamp;

    pc_publisher.publish(my_cloud);

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // ROS_INFO("[Local Mapping:] Merged Point cloud published, time %.4f\n", elapsed_seconds.count());
}

void PointProjectionNode::publishImages(uint8_t* colormap, float* heightmap,
                                     ros::Publisher& rgb_pub, ros::Publisher& height_pub, 
                                     ros::Time stamp, string frame_id)
{
    sensor_msgs::ImagePtr imgmsg, heightmsg;
    auto colorimg = cv::Mat(map_height, map_width, CV_8UC3, colormap);
    imgmsg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", colorimg).toImageMsg();
    imgmsg->header.stamp = stamp;
    imgmsg->header.frame_id = frame_id;
    rgb_pub.publish(*imgmsg);


    auto heightimg = cv::Mat(map_height, map_width, CV_32FC4, heightmap);
    heightmsg = cv_bridge::CvImage(std_msgs::Header(), "32FC4", heightimg).toImageMsg();
    heightmsg->header.stamp = stamp;
    heightmsg->header.frame_id = frame_id;
    height_pub.publish(*heightmsg);
}

void PointProjectionNode::handle_pc(const sensor_msgs::PointCloud2ConstPtr& pc_msg)
{
    auto start = std::chrono::steady_clock::now();
    pointcloud2_current = convertFromMsgToPointCloud(*pc_msg);

    if(align_gravity)
        alignGravity(pointcloud2_current, pointcloud2_current_ground, latest_ori);
    else
        pcl::transformPointCloud(pointcloud2_current, pointcloud2_current_ground, t_ground2cam); // convert from cam frame to ground frame

    if(publish_pc_for_debug)
        publishPointCloud(pointcloud2_current_ground, pc_msg->header.stamp, pc_msg->header.frame_id);
    
    // project the pointcloud to the 2d plane
    localmap.pc_to_map(pointcloud2_current_ground);
    if(visualize_maps)
    {
        localmap.show_color_map();
        localmap.show_height_map();
    }

    uint8_t* colormap = localmap.getColormap();
    float* heightmap = localmap.getHeightmap();
    publishImages(colormap, heightmap, rgb_publisher, height_publisher,
                    pc_msg->header.stamp, pc_msg->header.frame_id);

    if(inflate_fill_hole)
    {
        localmap.inflate_maps(3, false);
        for(int k=0;k<5;k++) // inflate the map multiple times to fill more holes
            localmap.inflate_maps(3, true);
        uint8_t* colormapinflate = localmap.getInflateColormap();
        float* heightmapinflate = localmap.getInflateHeightmap();
        publishImages(colormapinflate, heightmapinflate, rgb_inflate_publisher, height_inflate_publisher,
                    pc_msg->header.stamp, pc_msg->header.frame_id);
    }    

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_all = end - start;

    ROS_INFO_THROTTLE(1.0, "[Local projection] received %d points, mapping time %.4f", 
                        pc_msg->width * pc_msg->height, 
                        elapsed_seconds_all.count());

}

void PointProjectionNode::alignGravity(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, Eigen::Quaternionf quat)
{
    Eigen::Matrix3f rot_mat = quat.toRotationMatrix();
    Eigen::Vector3f xx, yy, zz;
    float norm;
    norm = sqrt(rot_mat(0,1) * rot_mat(0,1) + rot_mat(1,1) * rot_mat(1,1));
    yy << rot_mat(0,1)/norm, rot_mat(1,1)/norm, 0.0;
    zz << 0.0, 0.0, 1.0;
    xx = yy.cross(zz);

    Eigen::Matrix3f rot_mat2, rot_mat3;
    rot_mat2.block<3, 1>(0, 0) = xx.transpose();
    rot_mat2.block<3, 1>(0, 1) = yy.transpose();
    rot_mat2.block<3, 1>(0, 2) = zz.transpose();

    // rot_mat3 = rot_mat.transpose() * rot_mat2;
    rot_mat3 = rot_mat2.transpose() * rot_mat; // T_ground_to_novatel
    Eigen::Matrix4f t_rot_mat3 = Eigen::Matrix4f::Identity();;
    t_rot_mat3.block<3, 3>(0,0) = rot_mat3;

    Eigen::Matrix4f T;
    T = t_body2novatel * t_rot_mat3 * (t_body2novatel.transpose()) * t_ground2cam;
    pcl::transformPointCloud(input_pc, output_pc, T);

}


void PointProjectionNode::handle_odom(const nav_msgs::Odometry& odom_msg)
{
    latest_ori = Eigen::Quaternionf(odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z);
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_projection_node");
    PointProjectionNode local_mapping;

    ros::spin();
}