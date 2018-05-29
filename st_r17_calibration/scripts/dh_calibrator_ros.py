#!/usr/bin/python2

import numpy as np
import sympy
import rospy

import tf
import tf.transformations as tx
import functools

import message_filters
from approx_sync import ApproximateSynchronizer
from collections import deque, defaultdict

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection

from st_r17_calibration.dh_calibrator import DHCalibrator
# using ^ as placeholder for now

def fill_pose_msg(msg, txn, rxn):
    msg.position.x = txn[0]
    msg.position.y = txn[1]
    msg.position.z = txn[2]

    msg.orientation.x = rxn[0]
    msg.orientation.y = rxn[1]
    msg.orientation.z = rxn[2]
    msg.orientation.w = rxn[3]

""" ROS Binding """
class DHCalibratorROS(object):
    def __init__(self):
        # update params
        self._tfl = tf.TransformListener()
        self._step = 0
        self._last_update = 0
        self._update_step = 8
        self._start_step  = 64 * 4
        self._batch_size  = 64
        self._mem_size    = 64 * 64

        # ~dh = nominal DH parameter
        self._dh = rospy.get_param('~dh', default=None)
        if self._dh is None:
            raise ValueError('Improper DH Parameters Input : {}'.format(self._dh))

        self._dh = np.float32(self._dh)
        self._num_markers = rospy.get_param('~num_markers', default=1)
        self._noise = rospy.get_param('~noise', default=True)

        if self._noise:

            sc = np.float32([np.deg2rad(3.0), 0.03, 0.03, np.deg2rad(3.0)])

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

        # initial DH parameter
        self._dh0 = dh 
        self._dhf = None

        self._calib = DHCalibrator(dh, m=self._num_markers)
        self._calib.start()

        self._data = deque(maxlen = self._mem_size)
        self._Ys = [None for _ in range(self._num_markers)]

        self._sub = ApproximateSynchronizer(slop=0.001, fs=[
            message_filters.Subscriber('joint_states', JointState),
            message_filters.Subscriber('stereo_to_target', AprilTagDetectionArray)
            ], queue_size=20)
        self._gt_sub = rospy.Subscriber('base_to_target', AprilTagDetectionArray, self.ground_truth_cb) # fixed w.r.t. time

        self._sub.registerCallback(self.data_cb)
        self._pub = rospy.Publisher('dh', AprilTagDetectionArray, queue_size=10)
        self._vpub = rospy.Publisher('dh_viz', PoseArray, queue_size=10)
        self._m2i = defaultdict(lambda : len(self._m2i))
        self._i2m = {}
        self._seen = [False for _ in range(self._num_markers)]

        self._errs = []

    def ground_truth_cb(self, detection_msgs):
        for pm in detection_msgs.detections:
            # TODO : technically requires tf.transformPose(...) for robustness.
            # i.e. into base_link frame. for now, ok
            q = pm.pose.pose.pose.orientation
            p = pm.pose.pose.pose.position
            m_id = pm.id[0]
            Y = tx.compose_matrix(
                    angles = tx.euler_from_quaternion([q.x, q.y, q.z, q.w]),
                    translate = [p.x, p.y, p.z]
                    )
            i = self._m2i[m_id] # transform tag id to index
            if i >= self._num_markers:
                continue
            self._i2m[i] = m_id
            self._Ys[i] = Y

    def data_cb(self, joint_msg, detection_msgs):
        try:
            jorder = ['waist', 'shoulder', 'elbow', 'hand', 'wrist']
            j_idx = [joint_msg.name.index(j) for j in jorder]
            j = [joint_msg.position[i] for i in j_idx]
            j = np.concatenate( (j, [0]) ) # assume final joint is fixed
        except Exception as e:
            rospy.logerr_throttle(1.0, 'Joint Positions Failed : {} \n {}'.format(e, joint_msg))
        
        Xs = [np.eye(4) for _ in range(self._num_markers)]
        vis = [False for _ in range(self._num_markers)]

        if len(detection_msgs.detections) <= 0:
            return

        try:
            for pm in detection_msgs.detections:
                # pm.pose = pwcs
                # pm.pose.pose = pwc
                # pm.pose.pose.pose = p


                # TODO : technically requires tf.transformPose(...) for robustness.
                #pose = tf.transformPose(

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
                X = tx.compose_matrix(
                        angles = tx.euler_from_quaternion([q.x, q.y, q.z, q.w]),
                        translate = [p.x, p.y, p.z]
                        )
                i = self._m2i[m_id] # transform tag id to index
                if i >= self._num_markers:
                    continue
                self._i2m[i] = m_id
                Xs[i] = X
                vis[i] = True
                self._seen[i] = True
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))

        Xs = np.float32(Xs)

        self._data.append( (j, Xs, vis) )

        self._step += 1
        rospy.loginfo_throttle(1.0, 'Current Step : {}; vis : {}'.format(self._step, vis))
        
        T = self._calib.eval_1(j, Xs, vis) # == (1, M, 4, 4)
        
        now = rospy.Time.now()

        m_msg = AprilTagDetectionArray()
        m_msg.header.frame_id = 'base_link'
        m_msg.header.stamp = now

        pv_msg = PoseArray()
        pv_msg.header.stamp = now
        pv_msg.header.frame_id = 'base_link'

        for i in range(self._num_markers):
            if vis[i]:
                txn = tx.translation_from_matrix(T[i])
                rxn = tx.quaternion_from_matrix(T[i])

                msg = AprilTagDetection()
                msg.id = [self._i2m[i]]
                msg.size = [0.0] # not really a thing

                pwcs = msg.pose
                pwcs.header.frame_id = 'base_link'
                pwcs.header.stamp = now
                fill_pose_msg(pwcs.pose.pose, txn, rxn)

                m_msg.detections.append(msg)
                pv_msg.poses.append(pwcs.pose.pose)
        self._pub.publish(m_msg)
        self._vpub.publish(pv_msg)

    def update(self):
        if self._step < self._start_step:
            return
        if self._step < (self._last_update + self._update_step):
            return
        idx = np.random.randint(low=0, high = len(self._data), size = self._batch_size)
        js, Xs, vis = zip(*[self._data[i] for i in idx])

        for y in self._Ys:
            if y is None:
                rospy.loginfo_throttle(1.0, 'Not all target markers were initialized')
                return

        err, dhs = self._calib.step(js, Xs, vis, self._Ys)
        dhs = [dh[:3] for dh in dhs]
        real_err = np.mean(np.square(np.subtract(self._dh[:,:3], dhs)))

        self._last_update = self._step
        rospy.loginfo('Current Error : {} / {}'.format(err, real_err))
        rospy.loginfo_throttle(5.0, 'Current DH : {}'.format(dhs))

        # save data ...
        self._errs.append(real_err)
        self._dhf = dhs

    def save(self):
        np.savetxt('/tmp/err.csv', self._errs)
        np.savetxt('/tmp/dh0.csv', self._dh0)
        np.savetxt('/tmp/dhf.csv', self._dhf)

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.update()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        self.save()

def main():
    rospy.init_node('dh_calibrator')
    app = DHCalibratorROS()
    app.run()

if __name__ == "__main__":
    main()

