#!/usr/bin/python2

import numpy as np
import sympy
import rospy

import tf
import tf.transformations as tx
import functools

import message_filters
from collections import deque, defaultdict

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection

from st_r17_calibration.approx_sync import ApproximateSynchronizer

class DataCollector(object):
    def __init__(self):
        # params
        self._num_markers = rospy.get_param('~num_markers', default=4)
        self._slop = rospy.get_param('~slop', default=0.01)

        self._tfl = tf.TransformListener()
        self._m2i = defaultdict(lambda : len(self._m2i))
        self._i2m = {}
        self._data = []

        self._sub = ApproximateSynchronizer(slop=self._slop, fs=[
            message_filters.Subscriber('joint_states', JointState), # ~50hz, 0.02
            message_filters.Subscriber('stereo_to_target', AprilTagDetectionArray) # ~10hz, 0.1
            ], queue_size=20)
        self._sub.registerCallback(self.data_cb)

    def data_cb(self, joint_msg, detection_msgs):
        if len(detection_msgs.detections) <= 0:
            return

        # re-order joints
        try:
            jorder = ['waist', 'shoulder', 'elbow', 'hand', 'wrist']
            j_idx = [joint_msg.name.index(j) for j in jorder]
            j = [joint_msg.position[i] for i in j_idx]
            j = np.concatenate( (j, [0]) ) # assume final joint is fixed
        except Exception as e:
            rospy.logerr_throttle(1.0, 'Joint Positions Failed : {} \n {}'.format(e, joint_msg))

        obs = []

        # transform detections + add observations
        try:
            for pm in detection_msgs.detections:
                ps = PoseStamped(
                        header = pm.pose.header,
                        pose = pm.pose.pose.pose
                        )
                # this transform should theoretically be painless + static
                # Optionally, store the transform beforehand and apply it
                ps = self._tfl.transformPose('stereo_optical_link', ps)

                q = ps.pose.orientation
                p = ps.pose.position
                m_id = pm.id[0]
                i = self._m2i[m_id] # transform tag id to index
                if i >= self._num_markers:
                    continue
                self._i2m[i] = m_id
                obs.append([i, p,q])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))

        self._data.append([j, obs])

    def save(self):
        np.save('/tmp/dat.npy', self._data)

    def run(self):
        rate = rospy.Rate(50)
        try:
            while not rospy.is_shutdown():
                rate.sleep()
        except rospy.ROSInterruptException:
            self.save()

def main():
    app = DataCollector()
    app.run()

if __name__ == "__main__":
    main()
