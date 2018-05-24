#!/usr/bin/python2

import numpy as np
import sympy
import rospy
import tf.transformations as tx
import tensorflow as tf
import functools

import message_filters
from approx_sync import ApproximateSynchronizer
from collections import deque

from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState

eps=np.finfo(float).eps*4.0

def T2xyzrpy(T):
    with tf.name_scope('T2xyzrpy', [T]):
        # assume T = (N,4,4)
        i,j,k = (0,1,2)

        x = T[:,0,3]
        y = T[:,1,3]
        z = T[:,2,3]

        cy = tf.sqrt(
                tf.square(T[:,i, i]) + 
                tf.square(T[:,j, i])
                )

        eps_mask = tf.cast(tf.greater(cy, eps), tf.float32)

        ax0 = tf.atan2( T[:,k, j],  T[:,k, k])
        ay = tf.atan2(-T[:,k, i],  cy)
        az0 = tf.atan2( T[:,j, i],  T[:,i, i])

        ax1 = tf.atan2(-T[:, j, k],  T[:, j, j])
        az1 = tf.zeros_like(az0)

        ax = eps_mask * ax0 + (1.0 - eps_mask) * ax1 
        az = eps_mask * az0 + (1.0 - eps_mask) * az1 

        xyz = tf.stack([x,y,z], axis=-1)
        rpy = tf.stack([ax,ay,az], axis=-1)
    return xyz, rpy


""" Utility """
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

def get_dh(index, alpha0, a0, d0, q0=0):
    """ Seed DH Parameters; note q0 is offset joint value. """
    with tf.variable_scope('DH_{}'.format(index)):
        alpha = tf.Variable(dtype=tf.float32, initial_value=alpha0, trainable=True, name='alpha'.format(index))
        a = tf.Variable(dtype=tf.float32, initial_value=a0, trainable=True, name='a'.format(index))
        d = tf.Variable(dtype=tf.float32, initial_value=d0, trainable=True, name='d'.format(index))
        dq = tf.Variable(dtype=tf.float32, initial_value=q0, trainable=True, name='dq'.format(index))
        q = tf.placeholder(dtype=tf.float32, shape=[None], name='q'.format(index))
    return (alpha, a, d, q+dq), q

""" DH Calibrator """
class DHCalibrator(object):
    def __init__(self, dh0):
        self._dh0 = np.float32(dh0)
        self._initialized = False
        self._build()
        self._log()

    @staticmethod
    def _dh(index, alpha0, a0, d0, q0):
        """ Seed DH Parameters; note q0 is offset joint value. """
        with tf.variable_scope('DH_{}'.format(index)):
            alpha = tf.Variable(dtype=tf.float32, initial_value=alpha0, trainable=True, name='alpha'.format(index))
            a = tf.Variable(dtype=tf.float32, initial_value=a0, trainable=True, name='a'.format(index))
            d = tf.Variable(dtype=tf.float32, initial_value=d0, trainable=True, name='d'.format(index))
            dq = tf.Variable(dtype=tf.float32, initial_value=q0, trainable=True, name='dq'.format(index))
            q = tf.placeholder(dtype=tf.float32, shape=[None], name='q'.format(index))
        return (alpha, a, d, q+dq), q

    def _build(self):
        tf.reset_default_graph()
        self._graph = tf.get_default_graph()
        # inputs ...
        with tf.name_scope('inputs'):
            dhs, qs = zip(*[self._dh(i, *_dh) for (i, _dh) in enumerate(self._dh0)])
            T_f = tf.placeholder(dtype=tf.float32, shape=[None,4,4], name='T_f') # camera_link -> object

        # build  transformation ...
        with tf.name_scope('transforms'):
            Ts = [dh2T(*dh) for dh in dhs]
            Ts.append(T_f)
            T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> object

        mode = 'xyzrpy'
        pred_xyz, pred_rpy = T2xyzrpy(T)

        if mode == 'xyzrpy':
            with tf.name_scope('target_avg'):
                xyz_avg = tf.Variable(initial_value=[0,0,0], trainable=False, dtype=np.float32)
                rpy_avg = tf.Variable(initial_value=[0,0,0], trainable=False, dtype=np.float32)

                xyz_avg_1 = tf.reduce_mean(pred_xyz, axis=0)
                rpy_avg_1 = tf.reduce_mean(pred_rpy, axis=0)

                xyz0 = tf.assign(xyz_avg, xyz_avg_1)
                rpy0 = tf.assign(rpy_avg, rpy_avg_1)
                self._T_init = tf.group(xyz0, rpy0)

                gamma = 0.99

                xyz_avg_u = tf.assign(xyz_avg, gamma * xyz_avg + (1.0 - gamma) * xyz_avg_1)
                rpy_avg_u = tf.assign(rpy_avg, gamma * rpy_avg + (1.0 - gamma) * rpy_avg_1)
                T_avg_u = [xyz_avg_u, rpy_avg_u]
        else:
            with tf.name_scope('target_avg'):
                T_avg_1 = tf.reduce_mean(T, axis=0)
                T_avg = tf.Variable(initial_value = np.eye(4), trainable=False, dtype=np.float32)
                T_avg_u = [tf.assign(T_avg, 0.999 * T_avg + 0.001 * T_avg_1)]
                self._T_init = tf.assign(T_avg, T_avg_1) # don't forget to initialize!

        # tf.assert( ... )

        with tf.control_dependencies(T_avg_u + [tf.Print(xyz_avg, [xyz_avg])]):
            if mode == 'xyzrpy':
                loss_xyz = tf.square(pred_xyz - xyz_avg)
                loss_xyz = tf.reduce_mean(loss_xyz)
                loss_rpy = tf.square(pred_rpy - rpy_avg)
                loss_rpy = tf.reduce_mean(loss_rpy)
                loss = loss_xyz + loss_rpy
            else:
                loss = tf.reduce_mean(tf.square(T - tf.expand_dims(T_avg, 0)))

        # without running avg ... 
        #loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
        #loss = tf.reduce_mean(loss)

        # build train ...
        train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

        # save ...
        self._T = T
        self._T_f = T_f
        self._dhs = dhs
        self._qs = qs
        self._loss = loss
        if mode == 'xyzrpy':
            self._loss_xyz = loss_xyz
            self._loss_rpy = loss_rpy
        self._train = train

    def _log(self):
        # logs ...
        tf.summary.scalar('loss_xyz', self._loss_xyz)
        tf.summary.scalar('loss_rpy', self._loss_rpy)

        self._writer = tf.summary.FileWriter('/tmp/dh/11', self._graph)
        self._summary = tf.summary.merge_all()

    def eval_1(self, js, xs):
        feed_dict = {q:[j] for q,j in zip(self._qs, js)}
        feed_dict[self._T_f] = [xs]
        return self.run(self._T, feed_dict = feed_dict)[0]

    def start(self):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._iter = 0

    def step(self, js, xs):
        feed_dict = {q:j for q,j in zip(self._qs, np.transpose(js))}
        feed_dict[self._T_f] = xs

        if self._initialized:
            _, loss, dhs, summary = self.run(
                    [self._train, self._loss, self._dhs, self._summary],
                    feed_dict = feed_dict)
            self._writer.add_summary(summary, self._iter)
            self._iter += 1
            return loss, dhs
        else:
            _, dhs = self.run([self._T_init, self._dhs], feed_dict=feed_dict)
            self._initialized = True
            return -1, dhs

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

""" ROS Binding """
class DHCalibratorROS(object):
    def __init__(self):
        # update params
        self._step = 0
        self._last_update = 0
        self._update_step = 32
        self._start_step  = 32 * 32
        self._batch_size  = 128
        self._mem_size    = 32 * 64

        self._dh0 = [
                [np.pi, 0, -(0.033 + 0.322), 0],
                [np.pi/2, 0, 0, -np.pi/2],
                [0, 0.375, 0, 0],
                [0, 0.375, 0.024, np.pi/2],
                [np.pi/2, 0, 0.042, 1.176],
                [0, -0.012, 0.159, np.pi]
                ]
        self._dh0 = np.float32(self._dh0)
        z = np.random.normal(
                loc = 0.0,
                scale = [np.deg2rad(1.0), 0.01, 0.01, np.deg2rad(1.0)],
                size = np.shape(self._dh0)
                )
        dh0 = np.add(self._dh0, z)
        self._calib = DHCalibrator(dh0)
        self._calib.start()

        self._js = deque(maxlen = self._mem_size)
        self._Xs = deque(maxlen = self._mem_size)

        self._sub = ApproximateSynchronizer(slop=0.001, fs=[
            message_filters.Subscriber('st_r17/joint_states', JointState),
            message_filters.Subscriber('target_pose', PoseStamped)],
            queue_size=4)
        self._sub.registerCallback(self.data_cb)
        self._pub = rospy.Publisher('dh', PoseStamped, queue_size=10)

    def data_cb(self, joint_msg, pose_msg):
        j = np.concatenate( (joint_msg.position, [0]) ) # assume final joint is fixed
        q = pose_msg.pose.orientation
        p = pose_msg.pose.position
        X = tx.compose_matrix(
                angles = tx.euler_from_quaternion([q.x, q.y, q.z, q.w]),
                translate = [p.x, p.y, p.z]
                )
        self._js.append(j)
        self._Xs.append(X)
        self._step += 1
        rospy.loginfo_throttle(1.0, 'Current Step : {}'.format(self._step))
        
        T = self._calib.eval_1(j, X)
        txn = tx.translation_from_matrix(T)
        rxn = tx.quaternion_from_matrix(T)
        
        msg = PoseStamped()
        msg.header.frame_id = 'base_link'
        msg.header.stamp = rospy.Time.now()
        
        msg.pose.position.x = txn[0]
        msg.pose.position.y = txn[1]
        msg.pose.position.z = txn[2]
    
        msg.pose.orientation.x = rxn[0]
        msg.pose.orientation.y = rxn[1]
        msg.pose.orientation.z = rxn[2]
        msg.pose.orientation.w = rxn[3]
    
        self._pub.publish(msg)

    def update(self):
        if self._step < self._start_step:
            return
        if self._step < (self._last_update + self._update_step):
            return
        idx = np.random.randint(low=0, high = len(self._js), size = self._batch_size)
        js = [self._js[i] for i in idx] # (N, J)
        Xs = [self._Xs[i] for i in idx] # (N, 4, 4)

        err, dhs = self._calib.step(js, Xs)
        dhs = [dh[:3] for dh in dhs]
        real_err = np.mean(np.square(np.subtract(self._dh0[:,:3], dhs)))

        self._last_update = self._step
        rospy.loginfo('Current Error : {} / {}'.format(err, real_err))
        rospy.loginfo_throttle(5.0, 'Current DH : {}'.format(dhs))

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

def main():
    rospy.init_node('dh_calibrator')
    app = DHCalibratorROS()
    app.run()

#
#def build():
#
#
#    # T 0 -> 1 ( n-1 == 0, n = 1 )
#    # uses theta_1, d_1, alpha_0, a_0
#
#    # for T_0 -> T_1
#    # alpha_0 along x_0 from z_0 to z_1
#    # a_0 along x_0 from o_0 to o_1
#    # d_1 along z_1
#    # theta_1 along z_1
#
#    # full transformation
#    #get_dh(index, alpha0, a0, d0)
#
#    dh01, q0 = get_dh(0, np.pi,    0,        -(0.033+0.322)         ) # base_link -> link_1
#    dh12, q1 = get_dh(1, np.pi/2,  0,          0,         (-np.pi/2)) # 1 -> 2
#    dh23, q2 = get_dh(2, 0,        0.375,      0                    ) # 2 -> 3
#    dh34, q3 = get_dh(3, 0,        0.375, 0.024,        np.pi/2) # 3 -> 4
#    dh45, q4 = get_dh(4, np.pi/2,  0.0,        0.042,            1.176) # 4 -> 5
#    dh56, q5 = get_dh(5, 0,       -0.012,      0.159                ) # 5 -> camera
#
#    #dh0 = get_dh(0, np.pi/2, 0, 0.0) # link_1 -> link_2, i=0
#    #dh1 = get_dh(1, 0, 0, 0.0) # link_2 -> link_3
#    #dh2 = get_dh(2, 0.0, 0.375, 0) # link_3 -> link_4
#    #dh3 = get_dh(3, np.pi/2, 0.375, 0) # link_4 -> link_5
#    #dh4 = get_dh(4, np.pi/2, 0.042, 0.024) # link_5 -> camera_link
#    #dh5 = get_dh(5, 0.0, 0.012, 0.159) # link_5 -> camera_link
#
#    T_f = tf.placeholder(dtype=tf.float32, shape=[None,4,4], name='T_f') # camera_link -> object
#
#    # account for offsets here, if any
#    # notably, the offset at last joint
#
#    dhs = [dh01,dh12,dh23,dh34,dh45,dh56]
#    qs = [q0, q1, q2, q3, q4, q5]
#    Ts = [dh2T(*dh) for dh in dhs]
#    Ts.append(T_f)
#    T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> object
#    return T, qs, T_f
#
#def jcb(msg):
#    global txn, rxn, qs, T, T_f, sess
#
#    js = np.concatenate((msg.position, [0]) )
#
#    feed_dict = {q:[j] for (q,j) in zip(qs, js)}
#    feed_dict[T_f] = np.reshape(np.eye(4), (1,4,4))
#
#    _T = sess.run(T, feed_dict=feed_dict)
#    txn = tx.translation_from_matrix(_T[0])
#    rxn = tx.quaternion_from_matrix(_T[0])
#
#def main():
#    global txn, rxn, qs, T, T_f, sess
#    T, qs, T_f = build()
#    loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
#    loss = tf.reduce_mean(loss)
#    train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)
#    sess = tf.Session()
#
#    feed_dict = {q:[0] for q in qs}
#    feed_dict[T_f] = np.reshape(np.eye(4), (1,4,4))
#    sess.run(tf.global_variables_initializer())
#    _T = sess.run(T, feed_dict=feed_dict)
#    txn = tx.translation_from_matrix(_T[0])
#    rxn = tx.quaternion_from_matrix(_T[0])
#
#    rospy.init_node('dh_test')
#    sub = rospy.Subscriber('/st_r17/joint_states', JointState, jcb)
#    pub = rospy.Publisher('dh', PoseStamped, queue_size=10)
#    msg = PoseStamped()
#    msg.header.frame_id = 'base_link'
#
#
#    rate = rospy.Rate(50)
#    while not rospy.is_shutdown():
#        msg.pose.position.x = txn[0]
#        msg.pose.position.y = txn[1]
#        msg.pose.position.z = txn[2]
#
#        msg.pose.orientation.x = rxn[0]
#        msg.pose.orientation.y = rxn[1]
#        msg.pose.orientation.z = rxn[2]
#        msg.pose.orientation.w = rxn[3]
#
#        pub.publish(msg)
#        rate.sleep()


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
