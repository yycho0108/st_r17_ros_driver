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
from st_r17_calibration.kinematics import fk

def cat(*args):
    return np.concatenate(args, axis=0)

class DataCollector(object):
    def __init__(self):

        # parameters
        self._num_markers = rospy.get_param('~num_markers', default=4)
        self._slop = rospy.get_param('~slop', default=0.01)
        self._noise = rospy.get_param('~noise', default=0.0)
        self._dh = rospy.get_param('~dh', default=None)

        if self._dh is None:
            rospy.loginfo('DH is None, saving joint values directly')
        else:
            self._dh = np.asarray(self._dh)
            if self._noise > 0.0:
                z = np.random.normal(
                        loc = 0.0,
                        scale = self._noise,
                        size = np.shape(self._dh)
                        )
                z = np.clip(z, -2*self._noise, 2*self._noise)
                self._dh += z

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
                p = [p.x, p.y, p.z]
                q = [q.x, q.y, q.z, q.w]
                m_id = pm.id[0]
                i = self._m2i[m_id] # transform tag id to index
                if i >= self._num_markers:
                    continue
                self._i2m[i] = m_id
                obs.append([m_id, p, q])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))

        self._data.append([j, obs])
        rospy.loginfo_throttle(1.0, len(self._data))

    def save(self, as_g2o=True):
        rospy.loginfo('saving ... {}'.format(len(self._data)))
        omega = (1.0 / self._noise) * np.eye(6)
        oi = np.triu_indices(6)
        omega = omega[oi].ravel()
        if as_g2o:
            num_nodes = 1 + len(self._data)
            zi0 = num_nodes
            nodes = []
            edges = []
            nodes.append([0, [0, 0, 0, 0, 0, 0, 1]]) # base_link
            z_nodes = [None for _ in range(self._num_markers)]
            for (j, o) in self._data:
                ni = len(nodes) # current node index
                p, q = fk(self._dh, j)
                nodes.append([ni, cat(p,q)])
                edges.append([0, ni, cat(p,q)])
                for (zi, zp, zq) in o:
                    edges.append([ni, zi0 + zi, cat(zp,zq,omega)])
                    if z_nodes[zi] is None:
                        z_nodes[zi] = cat(zp,zq)
            for zi, zn in enumerate(z_nodes):
                nodes.append([zi0+zi, zn])

            with open('/tmp/joint_graph.g2o', 'w+') as f:
                for ni, n in nodes:
                    nx = " ".join(map(str, n))
                    f.write('VERTEX_SE3:QUAT {} {}\n'.format(ni, nx))
                for z0, z1, z in edges:
                    zx = " ".join(map(str, z))
                    f.write('EDGE_SE3:QUAT {} {} {}\n'.format(z0,z1,zx))
        else:
            np.save('/tmp/joint_graph.npy', self._data)

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        self.save(as_g2o=(self._dh is not None))

def main():
    rospy.init_node('data_collector')
    app = DataCollector()
    app.run()

if __name__ == "__main__":
    main()
