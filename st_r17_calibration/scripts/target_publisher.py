#!/usr/bin/env python2

import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection

import tf.transformations as tx
from fk import fk

def fill_pose_msg(msg, txn, rxn):
    msg.position.x = txn[0]
    msg.position.y = txn[1]
    msg.position.z = txn[2]

    msg.orientation.x = rxn[0]
    msg.orientation.y = rxn[1]
    msg.orientation.z = rxn[2]
    msg.orientation.w = rxn[3]

class SimpleTargetPublisher(object):
    def __init__(self):
        rospy.init_node('simple_target_publisher')

        # defaults
        dhs = [
                [np.pi, 0, -(0.033 + 0.322), 0],
                [np.pi/2, 0, 0, -np.pi/2],
                [0, 0.375, 0, 0],
                [0, 0.375, 0.024, np.pi/2],
                [np.pi/2, 0, 0.042, 1.176],
                [0, -0.012, 0.159, np.pi]
                ]

        self._dhs = np.float32(dhs)
        self._rate = rospy.get_param('~rate', default=50)
        self._zero = rospy.get_param('~zero', default=False)
        self._num_markers = rospy.get_param('~num_markers', default=1)
        self._p = rospy.get_param('~p', default=0.5) # visibility

        xyz = np.random.uniform(low = -1.0, high = 1.0, size=(self._num_markers, 3))
        rpy = np.random.uniform(low = -3.14, high = 3.14, size=(self._num_markers, 3))
        rpy[:,1] /= 2.0 # -1.57 ~ 1.57 valid range

        self._xyz = xyz
        self._rpy = rpy

        self._ppub = rospy.Publisher('/stereo_to_target', AprilTagDetectionArray, queue_size=10)
        self._pvpub = rospy.Publisher('/target_pose_viz', PoseArray, queue_size=10)

        self._gpub = rospy.Publisher('/base_to_target', AprilTagDetectionArray, queue_size=10)
        self._gvpub = rospy.Publisher('/ground_truth_viz', PoseArray, queue_size=10)

        self._jpub = rospy.Publisher('/st_r17/joint_states', JointState, queue_size=10)


    def publish(self):
        now = rospy.Time.now()

        jpos = np.random.uniform(low=-np.pi, high=np.pi, size=5)
        if self._zero:
            jpos *= 0
        txn, rxn = fk(self._dhs, np.concatenate( [jpos, [0]] ))

        ## base_link --> object
        M07 = [tx.compose_matrix(translate = xyz, angles=rpy) for (xyz, rpy) in zip(self._xyz, self._rpy)]

        ## base_link --> stereo_optical_link
        M06 = tx.compose_matrix(
                angles = tx.euler_from_quaternion(rxn),
                translate = txn) # T_0_6

        M06R = tx.quaternion_matrix(rxn)
        Rt =  M06R[:3,:3].T
        T  = -np.matmul(Rt, np.reshape(txn, [3,1]))
        M06i = np.zeros((4,4), dtype=np.float32)
        M06i[:3,:3] = Rt
        M06i[:3, 3] = T[:,0]
        M06i[ 3, 3] = 1.0

        m_msg = AprilTagDetectionArray()
        m_msg.header.stamp = now
        m_msg.header.frame_id = 'stereo_optical_link'

        pv_msg = PoseArray()
        pv_msg.header.stamp = now
        pv_msg.header.frame_id = 'stereo_optical_link'

        M67 = [np.matmul(M06i, M) for M in M07]

        for i in range(self._num_markers):
            if np.random.random() > self._p:
                continue
            M = M67[i]
            txn = tx.translation_from_matrix(M)
            rxn = tx.quaternion_from_matrix(M)
            
            msg = AprilTagDetection()
            msg.id = [i]
            msg.size = [0.0] # not really a thing

            pwcs = msg.pose
            pwcs.header.frame_id = 'stereo_optical_link'
            pwcs.header.stamp = now
            fill_pose_msg(pwcs.pose.pose, txn, rxn)

            m_msg.detections.append(msg)
            pv_msg.poses.append(pwcs.pose.pose)

        self._ppub.publish(m_msg)
        self._pvpub.publish(pv_msg)

        g_msg = AprilTagDetectionArray()
        g_msg.header.stamp = now
        g_msg.header.frame_id = 'base_link'

        gv_msg = PoseArray()
        gv_msg.header.stamp = now
        gv_msg.header.frame_id = 'base_link'

        for i in range(self._num_markers):
            M = M07[i]
            txn = tx.translation_from_matrix(M)
            rxn = tx.quaternion_from_matrix(M)

            msg = AprilTagDetection()
            msg.id = [i]
            msg.size = [0.0] # not really a thing

            pwcs = msg.pose
            pwcs.header.frame_id = 'base_link'
            pwcs.header.stamp = now
            fill_pose_msg(pwcs.pose.pose, txn, rxn)
            
            g_msg.detections.append(msg)
            gv_msg.poses.append(pwcs.pose.pose)

        self._gpub.publish(g_msg)
        self._gvpub.publish(gv_msg)

        #gmsg = PoseStamped()
        #gmsg.header.frame_id = 'base_link'
        #txn = tx.translation_from_matrix(M07)
        #rxn = tx.quaternion_from_matrix(M07)
        #gmsg.header.stamp = now
        #fill_pose_msg(gmsg.pose, txn, rxn)
        #self._gpub.publish(gmsg)

        jmsg = JointState()
        jmsg.header.stamp = now
        jmsg.header.frame_id = 'map' #??

        jmsg.name = ['waist', 'shoulder', 'elbow', 'hand', 'wrist']
        jmsg.position = jpos
        jmsg.velocity = np.zeros_like(jpos)
        jmsg.effort = np.zeros_like(jpos)
        self._jpub.publish(jmsg)
        
    def run(self):
        rate = rospy.Rate(self._rate)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()

def main():
    app = SimpleTargetPublisher()
    app.run()

if __name__ == "__main__":
    main()
