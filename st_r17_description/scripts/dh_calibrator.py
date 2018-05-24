#!/usr/bin/python2

import numpy as np
import sympy
import rospy
import tf.transformations as tx
import tensorflow as tf

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

def get_dh(index, alpha0, a0, d0, q0=0):
    with tf.variable_scope('DH_{}'.format(index)):
        alpha = tf.Variable(dtype=tf.float32, initial_value=alpha0, trainable=True, name='alpha'.format(index))
        a = tf.Variable(dtype=tf.float32, initial_value=a0, trainable=True, name='a'.format(index))
        d = tf.Variable(dtype=tf.float32, initial_value=d0, trainable=True, name='d'.format(index))
        q = tf.placeholder(dtype=tf.float32, shape=[None], name='q'.format(index))
    return (alpha, a, d, q+q0), q

def dh2T(alpha, a, d, q):
    with tf.name_scope('dh2T', [alpha, a, d, q]):
        # input q = placeholder (N,) ; all else variable (,)
        # output = (N, 4, 4)

        n = tf.shape(q)[0]

        z = tf.zeros_like(q)
        o = tf.ones_like(q)

        cq = tf.cos(q)
        sq = tf.sin(q)
        ca = o * tf.cos(alpha)
        sa = o * tf.sin(alpha)

        T = tf.reshape([
            cq, -sq, z, o*a,
            sq*ca, cq*ca, -sa, -sa*d,
            sq*sa, cq*sa, ca, ca*d,
            z, z, z, o], shape=(4,4,n)) # --> (4,4,N)

        T = tf.transpose(T, [2,0,1]) # --> (N,4,4)
    return T

def build():


    # T 0 -> 1 ( n-1 == 0, n = 1 )
    # uses theta_1, d_1, alpha_0, a_0

    # for T_0 -> T_1
    # alpha_0 along x_0 from z_0 to z_1
    # a_0 along x_0 from o_0 to o_1
    # d_1 along z_1
    # theta_1 along z_1

    # full transformation
    #get_dh(index, alpha0, a0, d0)

    dh01, q0 = get_dh(0, np.pi,    0,        -(0.033+0.322)         ) # base_link -> link_1
    dh12, q1 = get_dh(1, np.pi/2,  0,          0,         (-np.pi/2)) # 1 -> 2
    dh23, q2 = get_dh(2, 0,        0.375,      0                    ) # 2 -> 3
    dh34, q3 = get_dh(3, 0,        0.375, 0.024,        np.pi/2) # 3 -> 4
    dh45, q4 = get_dh(4, np.pi/2,  0.0,        0.042,            1.176) # 4 -> 5
    dh56, q5 = get_dh(5, 0,       -0.012,      0.159                ) # 5 -> camera

    #dh0 = get_dh(0, np.pi/2, 0, 0.0) # link_1 -> link_2, i=0
    #dh1 = get_dh(1, 0, 0, 0.0) # link_2 -> link_3
    #dh2 = get_dh(2, 0.0, 0.375, 0) # link_3 -> link_4
    #dh3 = get_dh(3, np.pi/2, 0.375, 0) # link_4 -> link_5
    #dh4 = get_dh(4, np.pi/2, 0.042, 0.024) # link_5 -> camera_link
    #dh5 = get_dh(5, 0.0, 0.012, 0.159) # link_5 -> camera_link

    T_f = tf.placeholder(dtype=tf.float32, shape=[None,4,4], name='T_f') # camera_link -> object

    # account for offsets here, if any
    # notably, the offset at last joint

    dhs = [dh01,dh12,dh23,dh34,dh45,dh56]
    qs = [q0, q1, q2, q3, q4, q5]
    Ts = [dh2T(*dh) for dh in dhs]
    Ts.append(T_f)
    T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> object
    return T, qs, T_f

def jcb(msg):
    global txn, rxn, qs, T, T_f, sess

    js = np.concatenate((msg.position, [0]) )

    feed_dict = {q:[j] for (q,j) in zip(qs, js)}
    feed_dict[T_f] = np.reshape(np.eye(4), (1,4,4))

    _T = sess.run(T, feed_dict=feed_dict)
    txn = tx.translation_from_matrix(_T[0])
    rxn = tx.quaternion_from_matrix(_T[0])

def main():
    global txn, rxn, qs, T, T_f, sess
    T, qs, T_f = build()
    loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
    loss = tf.reduce_mean(loss)
    train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
    sess = tf.Session()

    feed_dict = {q:[0] for q in qs}
    feed_dict[T_f] = np.reshape(np.eye(4), (1,4,4))
    sess.run(tf.global_variables_initializer())
    _T = sess.run(T, feed_dict=feed_dict)
    txn = tx.translation_from_matrix(_T[0])
    rxn = tx.quaternion_from_matrix(_T[0])

    rospy.init_node('dh_test')
    sub = rospy.Subscriber('/st_r17/joint_states', JointState, jcb)
    pub = rospy.Publisher('dh', PoseStamped, queue_size=10)
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'


    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        msg.pose.position.x = txn[0]
        msg.pose.position.y = txn[1]
        msg.pose.position.z = txn[2]

        msg.pose.orientation.x = rxn[0]
        msg.pose.orientation.y = rxn[1]
        msg.pose.orientation.z = rxn[2]
        msg.pose.orientation.w = rxn[3]

        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()

#class DHCalibrator(object):
#    def __init__(self):
#        pass
#
#    @staticmethod
#    def dh2T(alpha, a, d, q):
#        """ Convert DH Parameters to Transformation Matrix """
#        cq = cos(q)
#        sq = sin(q)
#        ca = cos(alpha)
#        sa = sin(alpha)
#
#        T = Matrix([
#            [cq, -sq, 0, a],
#            [sq*ca, cq*ca, -sa, -sa*d],
#            [sq*sa, cq*sa, ca, ca*d],
#            [0, 0, 0, 1]
#            ])
#        return T
#
#    def _build(self, 
#
#    def dhs2T(dhs):
#        Ts = [self.dh2T(*dh) for dh in dhs]
#        T = reduce(lambda a,b: simplify(a*b), Ts)
#        return T
#
#    def calibrator(dhs, qs, xs):
#        T = dhs2T(dhs)
#        psi = T.jacobian(q)
#
#def fk(q)
