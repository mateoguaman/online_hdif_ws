#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from learned_cost_map.msg import FloatStamped, BoolStamped
from racepak.msg import rp_controls
import numpy as np

class Buffer:
    '''Maintains a scrolling buffer to maintain a window of data in memory

    Args:
        buffer_size: 
            Int, number of data points to keep in buffer
    '''
    def __init__(self, buffer_size, padded=False, pad_val=None):
        self.buffer_size = buffer_size
        if not padded:
            self._data = []
            self.data = np.array(self._data)
        else:
            assert pad_val is not None, "For a padded array, pad_val cannot be None."
            self._data = [pad_val] * buffer_size
            self.data = np.array(self._data)
        
    def insert(self, data_point):
        self._data.append(data_point)
        if len(self._data) > self.buffer_size:
            self._data = self._data[1:]
        self.data = np.array(self._data)

    def get_data(self):
        return self.data

    def show(self):
        print(self.data)

class InterventionDetectionNode(object):
    def __init__(self, brake_topic='/controls', acc_thresh=10, buffer_size=None, end_period=10, near_stop_eps=0.1):
        self.acc_thresh = acc_thresh
        self.buffer_size = buffer_size
        self.end_period = end_period
        self.near_stop_eps = near_stop_eps

        # Set up subscribers
        # rospy.Subscriber(odom_topic, Odometry, self.handle_odom, queue_size=1)
        rospy.Subscriber(brake_topic, rp_controls, self.handle_brake, queue_size=1)

        # Set up publishers
        self.intervention = False
        self.intervention_publisher = rospy.Publisher('/intervention', BoolStamped, queue_size=10)

        # Set data buffer
        pad_val = Odometry()
        self.odom_freq = 50
        if buffer_size is None:
            self.buffer_size = int(1*self.odom_freq)
        else:
            self.buffer_size = buffer_size

        self.buffer = Buffer(self.buffer_size, padded=True, pad_val=0)

    def handle_odom(self, msg):
        print("Received odometry message")
        vel_x = msg.twist.twist.linear.x
        vel_y = msg.twist.twist.linear.y
        vel = np.linalg.norm(np.array([vel_x, vel_y]))

        self.buffer.insert(vel)
        vel_buffer = self.buffer.data
        accel = np.gradient(vel_buffer)

        intervention = self.intervention_detector(vel_buffer, accel)

        print(f"Publishing intervention: {intervention}")
        intervention_msg = BoolStamped()
        intervention_msg.header = msg.header
        intervention_msg.data = intervention
        self.intervention_publisher.publish(intervention_msg)
        # print("Published intervention!")

    def handle_brake(self, msg):
        print("Received controls message")
        brake_val = msg.brake

        brake = (brake_val > 200)

        # self.buffer.insert(vel)
        # vel_buffer = self.buffer.data
        # accel = np.gradient(vel_buffer)

        # intervention = self.intervention_detector(vel_buffer, accel)

        print(f"Publishing intervention: {brake}")
        intervention_msg = BoolStamped()
        intervention_msg.header = msg.header
        intervention_msg.data = brake
        self.intervention_publisher.publish(intervention_msg)
        # print("Published intervention!")


    def intervention_detector(self, vel_vector, acc_vector):
        # Find maximum acceleration: Hopefully this represents stopping because of an intervention
        max_acc = np.argmax(np.absolute(acc_vector))

        # Once we find a hard acceleration, find out whether that is followed by the vehicle stopping
        remaining_vel = vel_vector[max_acc:]
        if len(remaining_vel) < self.end_period:
            return False
        near_stop_count = np.sum((remaining_vel < self.near_stop_eps))
        if near_stop_count >= 0.5*len(remaining_vel):
            return True
        
        # We might want to have a detector of +, -, + or  -, +, - signs 

        return False


if __name__ == "__main__":
    rospy.init_node("intervention_publisher", log_level=rospy.INFO)
    rospy.loginfo("Initialized intervention_publisher node")
    node = InterventionDetectionNode(brake_topic='/controls', acc_thresh=10, buffer_size=None, end_period=10, near_stop_eps=0.1)
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        rate.sleep()
