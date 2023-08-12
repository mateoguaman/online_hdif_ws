#include "LocalMappingNode.h"

LocalMappingNode::LocalMappingNode(): nh_("~")
{
    bool got_pcl_topic = nh_.param<std::string>("pointcloud_topic", pointcloud_topic, "/warthog5/lidar_points");
    // nh_.getParam("pointcloud_topic", pointcloud_topic, "/ColorizePclNode/color_pcl_arl");
    nh_.param<std::string>("robot", robot, "warthog5");
    nh_.param<std::string>("pose_topic", pose_topic, "/warthog5/pose");
    nh_.param<std::string>("odom_topic", odom_topic, "/warthog5/odom");
    ROS_INFO("[Local Mapping] Pointcloud topic: ");
    std::cout << "Received? " << got_pcl_topic << ". Topic: " <<pointcloud_topic << std::endl;
    ROS_INFO("[Local Mapping] Pose topic: ");
    std::cout << pose_topic << std::endl;
    ROS_INFO("[Local Mapping] Odom topic: ");
    std::cout << odom_topic << std::endl;
    nh_.param<std::string>("pointcloud_topic", pointcloud_topic, "/deep_cloud");
    nh_.param<std::string>("pose_topic", pose_topic, "/tartanvo_pose");
    nh_.param("resolution", resolution, 0.02);
    nh_.param("max_x", max_x, 10.0);
    nh_.param("min_x", min_x, 0.0);
    nh_.param("max_y", max_y, 5.0);
    nh_.param("min_y", min_y, -5.0);
    nh_.param("downsample_pointcloud_before", downsample_pointcloud_before, false);
    nh_.param("downsample_pointcloud_after", downsample_pointcloud_after, true);
    nh_.param("downsample_leafsize_before", downsample_leafsize_before, 0.05);
    nh_.param("downsample_leafsize_after", downsample_leafsize_after, 0.02);
    nh_.param("remove_outlier_count", remove_outlier_count, 5);
    nh_.param("remove_outlier_thresh", remove_outlier_thresh, 1.5);
    nh_.param("filter_outliers", filter_outliers, false);
    nh_.param("downsample_pointcloud_after_thresh", downsample_pointcloud_after_thresh, 300000); // downsample the pc only when points number is big 
    nh_.param("max_points_in_cloud", max_points_in_cloud, 200000); // downsample the pc only when points number is big 
    nh_.param("inflate_fill_hole", inflate_fill_hole, true);
    nh_.param("publish_pc_for_debug", publish_pc_for_debug, true); // downsample the pc only when points number is big 
    nh_.param("visualize_maps", visualize_maps, false); // downsample the pc only when points number is big 
    nh_.param("align_gravity", align_gravity, true); // align the point cloud with gravity according to the odom message 
    nh_.param("pc_frame_skip", pc_frame_skip, 5);
    nh_.param("pc_buffer_size", pc_buffer_size, 10);
    nh_.param<std::string>("platform", platform, "yamaha"); // yamaha, racer, warthog

    localmap.init(resolution, min_x, max_x, min_y, max_y); // = ScrollGrid(resolution, min_x, max_x, min_y, max_y); 
    map_height = localmap.getMapHeight();
    map_width = localmap.getMapWidth();

    height_publisher = nh_.advertise<Image> ("/local_height_map", 2);
    rgb_publisher = nh_.advertise<Image>("/local_rgb_map", 2);
    if(inflate_fill_hole)
    {
        height_inflate_publisher = nh_.advertise<Image> ("/local_height_map_inflate", 2);
        rgb_inflate_publisher = nh_.advertise<Image>("/local_rgb_map_inflate", 2);        
    }

    pc_sync.subscribe(nh_, pointcloud_topic, 10);
    // trans_sync.subscribe(nh_, "/tartanvo_transform", 1);
    pose_sync.subscribe(nh_, pose_topic, 10);
    odom_subscriber = nh_.subscribe(odom_topic, 10, &LocalMappingNode::handle_odom, this);
    // odom_subscriber = nh_.subscribe("/warthog5/odom", 100, &LocalMappingNode::handle_odom, this);


    sync_.reset(new Sync(MySyncPolicy(100), pc_sync, pose_sync));
    sync_->registerCallback(boost::bind(&LocalMappingNode::process_data, this, _1, _2));

    first_msg_received = false;

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
                                            -0.0,  0.  , -1.0,   1.14,
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

    // pc_buffer = boost::circular_buffer<pcl::PointCloud<PointT>>(pc_buffer_size);
    // pc_tf_buffer = boost::circular_buffer<geometry_msgs::Pose>(pc_buffer_size);

    ROS_INFO ("Local Mapping node initialized.");

    pc_counter = 0;
}

LocalMappingNode::~LocalMappingNode()
{

}

// this is an inplace opration
void LocalMappingNode::downsamplePointCloud(pcl::PointCloud<PointT> &pointcloud, double leafsize)
{
    // auto start = std::chrono::steady_clock::now();
    // pcl::PointCloud<PointT> pointcloud_downsampled_pcl;
    int pcnum = pointcloud.width * pointcloud.height;
    //Now we will downsample the point cloud
    vg.setInputCloud (boost::make_shared<pcl::PointCloud<PointT>> (pointcloud));
    vg.setLeafSize (leafsize, leafsize, leafsize);
    vg.filter (pointcloud);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // ROS_INFO ("[Local Mapping] PointCloud downsampling: %d -> %d. time %.4f\n",pcnum,  pointcloud.width * pointcloud.height, elapsed_seconds.count());
}


pcl::PointCloud<PointT> LocalMappingNode::convertFromMsgToPointCloud(const sensor_msgs::PointCloud2& pointcloud_msg)
{
    // auto start = std::chrono::steady_clock::now();
    pcl::PointCloud<PointT> pointcloud_pcl; //, pointcloud_downsampled_pcl;
    pcl::fromROSMsg(pointcloud_msg, pointcloud_pcl);

    // STEP 01: Check if we should downsample the input cloud or not
    if(downsample_pointcloud_before == true)
    {
        downsamplePointCloud(pointcloud_pcl, downsample_leafsize_before);
    }
    // return pointcloud_downsampled_pcl;

    // STEP 02: Check if we should filter the outliers or not
    if(filter_outliers)
    {
        // Removing outliers
        sor.setInputCloud (boost::make_shared<pcl::PointCloud<PointT> >(pointcloud_pcl));
        sor.setMeanK (remove_outlier_count);
        sor.setStddevMulThresh (remove_outlier_thresh);
        sor.filter (pointcloud_pcl);
    }
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // ROS_INFO("[Local Mapping] convert ros msg to pointcloud: %.4f seconds", elapsed_seconds.count());

    return (pointcloud_pcl);
}

Eigen::Matrix4f LocalMappingNode::transform2Matrix(const geometry_msgs::TransformStamped trans_msg)
{
    Eigen::Vector3f trans(trans_msg.transform.translation.x, trans_msg.transform.translation.y, trans_msg.transform.translation.z);
    Eigen::Quaternionf rot(trans_msg.transform.rotation.w, trans_msg.transform.rotation.x, trans_msg.transform.rotation.y, trans_msg.transform.rotation.z);
    Eigen::Matrix3f r = rot.toRotationMatrix();
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = r.transpose();
    T.block<3,1>(0,3) = -trans;
    return T;
}

Eigen::Matrix4f LocalMappingNode::pose2Matrix(geometry_msgs::Pose pose)
{
    Eigen::Vector3f trans(pose.position.x, pose.position.y, pose.position.z);
    Eigen::Quaternionf rot(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Matrix3f r = rot.toRotationMatrix();
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = r;
    T.block<3,1>(0,3) = trans;
    return T;
}

// transform input_pc from pose_init to pose_target
void LocalMappingNode::transformPointCloud(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, 
                                            const geometry_msgs::Pose pose_init, const geometry_msgs::Pose pose_target)
{
    // auto start = std::chrono::steady_clock::now();

    Eigen::Matrix4f T_init = pose2Matrix(pose_init);
    Eigen::Matrix4f T_target = pose2Matrix(pose_target);
    
    Eigen::Matrix4f T = T_target.inverse() * T_init;
    pcl::transformPointCloud(input_pc, output_pc, T);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // ROS_INFO("[Local mapping node] coordinate transform time %.4f\n", elapsed_seconds.count());
}

// void LocalMappingNode::test_eigen_transform()
// {
//     MatrixXf m = MatrixXf::Random(3,50000);
//     MatrixXf mm = MatrixXf::Random(3,1);
//     Matrix3f t_test ;
//     t_test <<  0.9692535,   0.00610748, -0.24598853,  
//               0.,         -0.99969192, -0.02482067,
//             -0.24606434,   0.02405752,  -0.96895489;
//     auto start = std::chrono::steady_clock::now();
//     MatrixXf trans;
//     trans = t_test * m; // + mm;

//     auto end = std::chrono::steady_clock::now();
//     std::chrono::duration<double> elapsed_seconds = end - start;
//     ROS_INFO("[Eigen multiplication: %d, %d ] %.4f\n", trans.rows(), trans.cols(), elapsed_seconds.count());
// }


// this is an inplace opration
void LocalMappingNode::filterPointCloudByRange(pcl::PointCloud<PointT> &input_pc)
{
    // auto start = std::chrono::steady_clock::now();
    pcl::PassThrough<PointT> pass;
    pcl::PointCloud<PointT>::Ptr input_ptr = input_pc.makeShared();
    pass.setInputCloud (input_ptr);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_x - 2.0, max_x + 2.0);
    //pass.setFilterLimitsNegative (true);
    pass.filter (input_pc);

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // ROS_INFO("[Local mapping node] point filter time %.4f\n", elapsed_seconds.count());

}

pcl::PointCloud<PointT> LocalMappingNode::filterPointCloudByNumber(pcl::PointCloud<PointT> &input_pc)
{

    pcl::PointCloud<PointT> filtered;
    // ROS_INFO("%d, %d", input_pc.width, input_pc.height);
    filtered.width = max_points_in_cloud;
    filtered.height = 1;
    for(int i = 0; i < max_points_in_cloud; i++)
    {
        PointT point = input_pc.points[i];
        filtered.points.push_back(point);
    }
    return filtered;
    // pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    // pcl::ExtractIndices<PointT> extract;
    // for (int i = 0; i < 10000; i++)
    // {
    //     inliers->indices.push_back(i);
    // }
    // extract.setInputCloud(&input_pc);
    // extract.setIndices(inliers);
    // extract.setNegative(true);
    // extract.filter(input_pc);
}

void LocalMappingNode::publishPointCloud(const pcl::PointCloud<PointT> &pointcloud, ros::Time stamp, string frame_id_)
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

void LocalMappingNode::publishImages(uint8_t* colormap, float* heightmap,
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


// The point filtering is very slow: downsamplePointCloud, StatisticalOutlierRemoval takes a lot of time
// But without filtering, the number of points become large, causing the global2ground and pc2map slow 
// 1 Million points: 
//   convertFromMsgToPointCloud (downsamplePointCloud + outlierRemoval included, about 100k new points): 0.08s
//   registration (downsamplePointCloud included): 0.11s
//   gloabel2ground: 0.02s 
//   pc2map: 0.02s
void LocalMappingNode::process_data(const sensor_msgs::PointCloud2ConstPtr& pc_msg, 
                const geometry_msgs::PoseStampedConstPtr &pose_msg)
{
    pc_counter ++;

    auto start = std::chrono::steady_clock::now();
    pointcloud2_current = convertFromMsgToPointCloud(*pc_msg);

    // auto start1 = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = start1 - start;
    // ROS_INFO("[Local mapping node] convert msg time %.4f", elapsed_seconds.count());

    last_pose = current_pose;
    current_pose = pose_msg->pose;
    if(!first_msg_received)
    {
        first_msg_received = true;
        pointcloud2_merged = pointcloud2_current;
    }
    else
    {
        // Eigen::Matrix4f trans_matrix = transform2Matrix(*trans_msg);
        pcl::PointCloud<PointT> pc_global_trans;
        transformPointCloud(pointcloud2_merged, pc_global_trans, last_pose, current_pose);

        pointcloud2_merged = pointcloud2_current + pc_global_trans;
        int merged_pt_count = pointcloud2_merged.width * pointcloud2_merged.height;
        if( downsample_pointcloud_after == true && merged_pt_count > downsample_pointcloud_after_thresh)
        {
            downsamplePointCloud(pointcloud2_merged, downsample_leafsize_after);
        }
    }

    // auto start2 = std::chrono::steady_clock::now();
    // elapsed_seconds = start2 - start1;
    // ROS_INFO("[Local mapping node] registration time %.4f", elapsed_seconds.count());

    filterPointCloudByRange(pointcloud2_merged);
    if(pointcloud2_merged.width > max_points_in_cloud)
        pointcloud2_merged = filterPointCloudByNumber(pointcloud2_merged);

    if(align_gravity)
        alignGravity(pointcloud2_merged, pointcloud2_merged_ground, latest_ori);
    else
        pcl::transformPointCloud(pointcloud2_merged, pointcloud2_merged_ground, t_ground2cam); // convert from cam frame to ground frame

    // auto start3 = std::chrono::steady_clock::now();
    // elapsed_seconds = start3 - start2;
    // ROS_INFO("[Local mapping node] global2ground %d time %.4f", pointcloud2_merged.width*pointcloud2_merged.height, elapsed_seconds.count());

    if(publish_pc_for_debug)
        publishPointCloud(pointcloud2_merged_ground, pc_msg->header.stamp, pc_msg->header.frame_id);
    
    // test_eigen_transform();

    // project the pointcloud to the 2d plane
    localmap.pc_to_map(pointcloud2_merged_ground);
    if(visualize_maps)
    {
        localmap.show_color_map();
        localmap.show_height_map();
    }

    // auto start4 = std::chrono::steady_clock::now();
    // elapsed_seconds = start4 - start3;
    // ROS_INFO("[Local mapping node] pc to map time %.4f", elapsed_seconds.count());

    uint8_t* colormap = localmap.getColormap();
    float* heightmap = localmap.getHeightmap();
    publishImages(colormap, heightmap, rgb_publisher, height_publisher,
                    pc_msg->header.stamp, pc_msg->header.frame_id);

    if(inflate_fill_hole)
    {
        localmap.inflate_maps(3, false);
        uint8_t* colormapinflate = localmap.getInflateColormap();
        float* heightmapinflate = localmap.getInflateHeightmap();
        publishImages(colormapinflate, heightmapinflate, rgb_inflate_publisher, height_inflate_publisher,
                    pc_msg->header.stamp, pc_msg->header.frame_id);
    }    

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds_all = end - start;

    ROS_INFO_THROTTLE(1.0, "[Local mapping] received %d points, after registration %d, mapping time %.4f", 
                        pc_msg->width * pc_msg->height, 
                        pointcloud2_merged_ground.width * pointcloud2_merged_ground.height,
                        elapsed_seconds_all.count());

}

void LocalMappingNode::alignGravity(const pcl::PointCloud<PointT> &input_pc, pcl::PointCloud<PointT> &output_pc, Eigen::Quaternionf quat)
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


void LocalMappingNode::handle_odom(const nav_msgs::Odometry& odom_msg)
{
    latest_ori = Eigen::Quaternionf(odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z);
    // Eigen::Vector3f euler = r.eulerAngles(0,1,2);
    // ROS_INFO("%.4f, %.4f, %.4f", euler[0]*180/M_PI, euler[1]*180/M_PI, euler[2]*180/M_PI);
    // Eigen::Vector3f euler2 = rot_mat.eulerAngles(2,1,0);
    // ROS_INFO("%.4f, %.4f, %.4f", euler2[2]*180/M_PI, euler2[1]*180/M_PI, euler2[0]*180/M_PI);

    // std::cout<<"roll is " <<180/M_PI*atan2( r(2,1),r(2,2) ) <<std::endl;
    // std::cout<<"pitch is " <<180/M_PI*atan2( -r(2,0), std::pow( r(2,1)*r(2,1) +r(2,2)*r(2,2) ,0.5  )  ) <<std::endl;
    // std::cout<<"yaw is " <<180/M_PI*atan2( r(1,0),r(0,0) ) <<std::endl;
}

// void LocalMappingNode::handle_pc(const sensor_msgs::PointCloud2& pc_msg)
// {
//     // pointcloud2_current = convertFromMsgToPointCloud(pc_msg);

//     printf("[Local mapping node] test call back");
//     ros::Time pc_time = pc_msg.header.stamp;
//     for(TransformStamped trans : trans_buf)
//     {
//         ros::Time trans_time = trans.header.stamp;
//         ROS_INFO("%s, %s", pc_time, trans_time);
//         break;
//     }

// }

// void LocalMappingNode::handle_trans(const geometry_msgs::TransformStamped& trans_msg)
// {
//     // pointcloud2_current = convertFromMsgToPointCloud(pc_msg);

//     printf("[Local mapping node] test2 call back\n");
//     trans_buf.push_back(trans_msg);
// }

int main(int argc, char** argv)
{
    ros::init(argc, argv, "local_mapping_node");
    LocalMappingNode local_mapping;

    ros::spin();
}
