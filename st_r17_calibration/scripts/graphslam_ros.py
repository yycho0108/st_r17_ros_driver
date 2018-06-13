#!/usr/bin/env python2
import rospy
import numpy as np
import tf
import tf.transformations as tx

import message_filters
from collections import deque, defaultdict
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection

from st_r17_calibration.approx_sync import ApproximateSynchronizer
from st_r17_calibration.kinematics import fk
from st_r17_calibration.graphslam import GraphSlam3
from st_r17_calibration import qmath_np

def pmsg2pq(msg):
    p = msg.position
    q = msg.orientation
    return [p.x, p.y, p.z], [q.x,q.y,q.z,q.w]

def pose(p, q):
    msg = Pose()
    msg.position.x = p[0]
    msg.position.y = p[1]
    msg.position.z = p[2]
    msg.orientation.x = q[0]
    msg.orientation.y = q[1]
    msg.orientation.z = q[2]
    msg.orientation.w = q[3]
    return msg

def msg1(p, q, t, frame='base_link'):
    msg = PoseStamped()
    msg.pose = pose(p,q)
    msg.header.frame_id = frame
    msg.header.stamp = t
    return msg

def msgn(pqs, t, frame='base_link'):
    msg = PoseArray()
    msg.header.frame_id = frame
    msg.header.stamp = t
    msg.poses = [pose(p,q) for (p,q) in pqs]
    return msg

class GraphSlamROS(object):
    def __init__(self):
        self._tfl = tf.TransformListener()

        self._num_markers = rospy.get_param('~num_markers', default=4)
        self._noise = rospy.get_param('~noise', default=True)
        self._slop = rospy.get_param('~slop', default=0.01)

        self._dh = rospy.get_param('~dh', default=None)
        if self._dh is None:
            raise ValueError('Improper DH Parameters Input : {}'.format(self._dh))
        self._dh = np.float32(self._dh)

        # add noise to dh ... or else.
        if self._noise:
            sc = 1.0 * np.float32([np.deg2rad(1.0), 0.01, 0.01, np.deg2rad(1.0)])
            # for testing with virtual markers, etc.
            z = np.random.normal(
                    loc = 0.0,
                    scale = sc,
                    size = np.shape(self._dh)
                    )
            z = np.clip(z, -2*sc, 2*sc)
            dh = np.add(self._dh, z)
        else:
            # for testing for real!
            dh = self._dh.copy()
            
        self._m2i = defaultdict(lambda : len(self._m2i))
        self._i2m = {}

        self._dh0 = dh
        self._initialized = False
        self._slam = GraphSlam3(n_l=self._num_markers,
                l = 10000.0 # marquardt parameter
                )

        # default observation fisher information
        # TODO : configure, currently just a guess
        cov = np.square([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        cov = np.divide(1.0, cov)
        omega = np.diag(cov)
        self._omega = omega
        #self._omega = np.eye(6)


        self._pub_ee = rospy.Publisher('end_effector_est', PoseStamped, queue_size=10)
        self._pub_ze = rospy.Publisher('landmark_est', PoseArray, queue_size=10)

        self._sub = ApproximateSynchronizer(slop=self._slop, fs=[
            message_filters.Subscriber('joint_states', JointState), # ~50hz, 0.02
            message_filters.Subscriber('stereo_to_target', AprilTagDetectionArray) # ~10hz, 0.1
            ], queue_size=20)

        self._sub.registerCallback(self.data_cb)

    def data_cb(self, joint_msg, detection_msgs):
        # set info-mat param


        # reorder joint information
        try:
            jorder = ['waist', 'shoulder', 'elbow', 'hand', 'wrist']
            j_idx = [joint_msg.name.index(j) for j in jorder]
            j = [joint_msg.position[i] for i in j_idx]
            #j = np.random.normal(loc=j, scale=0.005)
            j = np.concatenate( (j, [0]) ) # assume final joint is fixed
        except Exception as e:
            rospy.logerr_throttle(1.0, 'Joint Positions Failed : {} \n {}'.format(e, joint_msg))

        if len(detection_msgs.detections) <= 0:
            # nothing detected, stop
            return

        # form pose ...
        p, q = fk(self._dh0, j)

        if not self._initialized:
            # save previous pose and don't proceed
            self._p = p
            self._q = q
            self._slam.initialize(np.concatenate([p,q], axis=0))
            self._initialized = True
            return

        # form observations ...
        zs = []
        try:
            for pm in detection_msgs.detections:
                m_id = pm.id[0]
                ps = PoseStamped(
                        header = pm.pose.header,
                        pose = pm.pose.pose.pose
                        )
                # TODO : compute+apply static transform?
                ps2 = self._tfl.transformPose('stereo_optical_link', ps)
                psp, psq = pmsg2pq(ps.pose)
                ps2p,ps2q = pmsg2pq(ps2.pose)
                zp, zq = ps.pose.position, ps.pose.orientation
                zp = [zp.x, zp.y, zp.z]
                zq = [zq.x,zq.y,zq.z,zq.w]
                # testing; add noise
                #zp = np.random.normal(zp, scale=0.01)
                #dzq = qmath_np.rq(s=0.01)
                #zq = qmath_np.qmul(dzq, zq)
                dz = np.concatenate([zp,zq], axis=0)
                zi = self._m2i[m_id]
                if zi >= self._num_markers:
                    continue
                self._i2m[zi] = m_id
                zs.append([1, 2+zi, dz, self._omega.copy()])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))
            return

        dp, dq = qmath_np.xrel(self._p, self._q, p, q)
        dx     = np.concatenate([dp,dq], axis=0)

        # save p-q
        self._p = p
        self._q = q

        # form dz ...
        mu = self._slam.step(x=dx, zs=zs)
        mu = np.reshape(mu, [-1, 7])
        ep, eq = qmath_np.x2pq(mu[1])
        ez     = [qmath_np.x2pq(e) for e in mu[2:]]
        #ez     = [unparametrize(e) for e in mu[1:]]

        t = joint_msg.header.stamp
        self._pub_ee.publish(msg1(ep, eq, t))
        self._pub_ze.publish(msgn(ez, t))
        
    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


def main():
    rospy.init_node('graph_slam')
    app = GraphSlamROS()
    app.run()

if __name__ == "__main__":
    main()
