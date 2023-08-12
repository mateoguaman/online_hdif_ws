import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R

def quat2SE(quat_data):
    '''
    quat_data: 7
    SE: 4 x 4
    '''
    SO = R.from_quat(quat_data[3:7]).as_dcm()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    return SE

def SE2quat(SE_data):
    '''
    SE_data: 4 x 4
    quat: 7
    '''
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def SO2quat(SO_data):
    rr = R.from_dcm(SO_data)
    return rr.as_quat()

def pose2motion(pose1, pose2, skip=0):
    '''
    pose1, pose2: [x, y, z, rx, ry, rz, rw]
    return motion: [x, y, z, rx, ry, rz, rw]
    '''
    pose1_SE = quat2SE(pose1)
    pose2_SE = quat2SE(pose2)
    motion = np.matrix(pose1_SE).I*np.matrix(pose2_SE)
    return SE2quat(motion)


class MushrSensingNode(object):
    def __init__(self):

        rospy.Subscriber('/mushr_mocap/odom', Odometry, self.handle_odom, queue_size=1)
        self.trans_pub = rospy.Publisher("/mushr_sensing/transform", TransformStamped, queue_size=10)
        self.last_pose = None


    def handle_odom(self, msg):
        cur_pose = [msg.pose.pose.position.x, 
                    msg.pose.pose.position.y, 
                    msg.pose.pose.position.z,
                    msg.pose.pose.orientation.x, 
                    msg.pose.pose.orientation.y, 
                    msg.pose.pose.orientation.z, 
                    msg.pose.pose.orientation.w]

        # print(cur_pose)
        if self.last_pose is not None:
            motion = pose2motion(self.last_pose, cur_pose)
            trans_msg = TransformStamped()
            trans_msg.header.stamp = msg.header.stamp
            trans_msg.header.frame_id = 'mushr'
            trans_msg.transform.translation.x = motion[0]
            trans_msg.transform.translation.y = motion[1]
            trans_msg.transform.translation.z = motion[2]
            trans_msg.transform.rotation.x = motion[3]
            trans_msg.transform.rotation.y = motion[4]
            trans_msg.transform.rotation.z = motion[5]
            trans_msg.transform.rotation.w = motion[6]
            self.trans_pub.publish(trans_msg)

        self.last_pose = cur_pose
        
if __name__ == '__main__':
    rospy.init_node("mushr_sensing_node", log_level=rospy.INFO)
    node = MushrSensingNode()
    rospy.loginfo("mushr_sensing node initialized")
    rospy.spin()
