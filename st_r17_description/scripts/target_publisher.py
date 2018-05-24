#!/usr/bin/env python2

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import tf.transformations as tx
from fk import fk

class SimpleTargetPublisher(object):
    def __init__(self):
        rospy.init_node('simple_target_publisher')

        self._dhs = [
                [np.pi, 0, -(0.033 + 0.322), 0],
                [np.pi/2, 0, 0, -np.pi/2],
                [0, 0.375, 0, 0],
                [0, 0.375, 0.024, np.pi/2],
                [np.pi/2, 0, 0.042, 1.176],
                [0, -0.012, 0.159, np.pi]
                ]
        self._dhs = np.float32(self._dhs)

        self._rate = rospy.get_param('~rate', default=50)
        self._zero = rospy.get_param('~zero', default=False)
        self._ppub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        self._jpub = rospy.Publisher('/st_r17/joint_states', JointState, queue_size=10)

    def publish(self):
        now = rospy.Time.now()
        pmsg = PoseStamped()

        x = np.random.uniform(low=-np.pi, high=np.pi, size=5)
        if self._zero:
            x *= 0
        txn, rxn = fk(self._dhs, np.concatenate( [x, [0]] ))

        ## base_link --> object
        #M07 = tx.compose_matrix(
        #        translate = (0, 0, 0)
        #        )# T_0_7
        M07 = np.eye(4)
        M06 = tx.compose_matrix(
                angles = tx.euler_from_quaternion(rxn),
                translate = txn) # T_0_6

        # inverse ... 

        # src = T_0_6
        # trg = T_0_7

        # 1 : M67
        pmsg.header.frame_id = 'stereo_optical_link'
        M67 = np.matmul(np.linalg.inv(M06), M07)
        txn = tx.translation_from_matrix(M67)
        rxn = tx.quaternion_from_matrix(M67)

        # 2 : M06
        # pmsg.header.frame_id = 'base_link'
        # txn = tx.translation_from_matrix(M06)
        # rxn = tx.quaternion_from_matrix(M06)

        #R = tx.quaternion_matrix(rxn)
        #Rt = R.T
        #M06i = np.copy(Rt)
        #M06i[:3, 3] = -np.matmul(Rt[:3,:3], np.reshape(txn, (3,1)))[:, 0]

        #M67 = np.matmul(M06i, M07)


        #R = tx.rotation_matrix(tx.euler_from_quaternion(rxn))


        #print np.linalg.inv(M)
        #print Mi

        #M = np.matmul(np.linalg.inv(M), M1)

        #pmsg.header.frame_id = 'stereo_link'
        #pmsg.header.frame_id = 'base_link'
        pmsg.header.stamp = now

        pmsg.pose.position.x = txn[0]
        pmsg.pose.position.y = txn[1]
        pmsg.pose.position.z = txn[2]
    
        pmsg.pose.orientation.x = rxn[0]
        pmsg.pose.orientation.y = rxn[1]
        pmsg.pose.orientation.z = rxn[2]
        pmsg.pose.orientation.w = rxn[3]

        self._ppub.publish(pmsg)

        jmsg = JointState()
        jmsg.header.stamp = now
        jmsg.header.frame_id = 'map' #??

        jmsg.name = ['waist', 'shoulder', 'elbow', 'hand', 'wrist']
        jmsg.position = x
        jmsg.velocity = np.zeros_like(x)
        jmsg.effort = np.zeros_like(x)
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
