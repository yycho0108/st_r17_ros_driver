#!/usr/bin/env python2

import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, PoseArray
from sensor_msgs.msg import JointState
from apriltags2_ros.msg import AprilTagDetectionArray, AprilTagDetection

from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelStateResponse

import tf
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

def get_xyz_rpy(size,
        min_z=0.1,
        min_r=3.0, max_r=5.0,
        min_phi=np.deg2rad(10), max_phi=np.deg2rad(30),
        min_theta=-np.pi, max_theta=np.pi):
    
    center = (0, 0, min_z)

    r = np.random.uniform(low=min_r, high=max_r, size=size)

    phi = np.random.uniform(low=min_phi, high=max_phi, size=size)

    theta = np.random.uniform(low=min_theta, high=max_theta, size=size)

    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)

    R = 0 * np.random.uniform(low=-np.pi, high=np.pi, size=size)
    P = -phi
    Y = theta

    xyz = np.stack([x,y,z], axis=-1)
    rpy = np.stack([R,P,Y], axis=-1)
    return xyz, rpy

class GazeboTargetInitializer(object):
    def __init__(self):
        rospy.init_node('gazebo_target_publisher')
        self._rate = rospy.get_param('~rate', default=50)
        self._zero = rospy.get_param('~zero', default=False)
        self._num_markers = rospy.get_param('~num_markers', default=1)
        self._tag_size = rospy.get_param('~tag_size', default=0.1)

        self._min_Y = rospy.get_param('~min_Y', default=-np.pi)
        self._max_Y = rospy.get_param('~max_Y', default=np.pi)
        self._min_P = rospy.get_param('~min_P', default=np.deg2rad(10))
        self._max_P = rospy.get_param('~max_P', default=np.deg2rad(30))

        xyz, rpy = get_xyz_rpy(self._num_markers,
                min_z = self._tag_size,
                min_phi = self._min_P,
                max_phi = self._max_P,
                min_theta = self._min_Y,
                max_theta = self._max_Y
                )
        self._xyz = xyz
        self._rpy = rpy

        rospy.wait_for_service('/gazebo/set_model_state')

        self._tfb = tf.TransformBroadcaster()
        self._gsrv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._gpub = rospy.Publisher('/base_to_target', AprilTagDetectionArray, queue_size=10)
        self._gvpub = rospy.Publisher('/ground_truth_viz', PoseArray, queue_size=10)

        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        try:
            for i in range(self._num_markers):
                #M = tx.compose_matrix(translate = self._xyz[i], angles=self._rpy[i])
                #txn = tx.translation_from_matrix(M)
                #rxn = tx.quaternion_from_matrix(M)
                txn = self._xyz[i]
                rxn = tx.quaternion_from_euler(*(self._rpy[i]))

                req = SetModelStateRequest()
                req.model_state.model_name = 'a_{}'.format(i)
                req.model_state.reference_frame = 'map' # TODO : wait for base_link for correctness. for now, ok.
                fill_pose_msg(req.model_state.pose, txn, rxn)
                res = self._gsrv(req)

                if not res.success:
                    rospy.logerr_throttle(1.0, 'Set Model Request Failed : {}'.format(res.status_message))
                    return

        except rospy.ServiceException as e:
            rospy.logerr_throttle(1.0, 'Initialization Failed : {}'.format(e))
            return

        self._initialized = True

    def publish(self):
        if not self._initialized:
            return
        now = rospy.Time.now()

        g_msg = AprilTagDetectionArray()
        g_msg.header.stamp = now
        g_msg.header.frame_id = 'base_link'

        gv_msg = PoseArray()
        gv_msg.header.stamp = now
        gv_msg.header.frame_id = 'base_link'

        # alignment ...
        q = tx.quaternion_from_euler(np.pi/2, 0, -np.pi/2) 

        for i in range(self._num_markers):
            txn = self._xyz[i]
            rxn = tx.quaternion_from_euler(*(self._rpy[i]))
            rxn = tx.quaternion_multiply(rxn, q)

            msg = AprilTagDetection()
            msg.id = [i]
            msg.size = [self._tag_size]

            pwcs = msg.pose
            pwcs.header.frame_id = 'base_link'
            pwcs.header.stamp = now
            fill_pose_msg(pwcs.pose.pose, txn, rxn)
            
            g_msg.detections.append(msg)
            gv_msg.poses.append(pwcs.pose.pose)
            self._tfb.sendTransform(txn, rxn, now,
                    child='gt_tag_{}'.format(i),
                    parent='base_link')

        self._gpub.publish(g_msg)
        self._gvpub.publish(gv_msg)
        
    def run(self):
        rate = rospy.Rate(self._rate)
        while not rospy.is_shutdown():
            self.initialize()
            self.publish()
            rate.sleep()

def main():
    app = GazeboTargetInitializer()
    app.run()

if __name__ == "__main__":
    main()
