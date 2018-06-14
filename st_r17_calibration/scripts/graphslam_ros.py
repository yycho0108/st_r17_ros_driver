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
        self._graph = deque(maxlen=1000)
        self._slam = GraphSlam3(n_l=self._num_markers,
                l = 10.0 # marquardt parameter
                )

        # default observation fisher information
        # TODO : configure, currently just a guess

        #cov = np.square([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        #cov = np.divide(1.0, cov)
        #omega = np.diag(cov)
        #self._omega = omega
        omega = np.diag([1,1,1,1,1,1])
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
        p_gt, q_gt = fk(self._dh, j)

        # testing ; add motion noise
        # p = np.random.normal(p, scale=0.01)
        # dq = qmath_np.rq(s=0.01)
        # q = qmath_np.qmul(dq, q)

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
                ps = self._tfl.transformPose('stereo_optical_link', ps)
                zp, zq = pmsg2pq(ps.pose)

                # testing; add noise
                # zp = np.random.normal(zp, scale=0.01)
                # dzq = qmath_np.rq(s=0.01)
                # zq = qmath_np.qmul(dzq, zq)

                dz = np.concatenate([zp,zq], axis=0)
                zi = self._m2i[m_id]
                if zi >= self._num_markers:
                    continue
                self._i2m[zi] = m_id
                zs.append([1, 2+zi, dz, self._omega.copy()])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))
            return

        #semi-offline batch optimization
        #self._graph.append([p, q, zs])
        #if len(self._graph) > 200:
        #    n_poses = len(self._graph)
        #    nodes = []
        #    edges = []
        #    z_nodes = [[] for _ in range(self._num_markers)]

        #    for g in self._graph:
        #        xi = len(nodes) # pose index

        #        gp, gq, gz = g
        #        gx = np.concatenate([gp,gq], axis=0)
        #        nodes.append(gx)
        #        # TODO : handle scenarios when a landmark
        #        # didn't appear in 200 observations
        #        for _, zi, dz, zo in gz:
        #            zi_ = n_poses + (zi-2) # re-computed landmark index
        #            edges.append([xi,zi,dz,zo])
        #            # compute absolute position to start from good-ish mean
        #            zp_a, zq_a = qmath_np.x2pq(qmath_np.xadd_rel(gx, dz, T=False))
        #            z_nodes[zi-2].append([zp_a, zq_a])

        #    for zi, zn in enumerate(z_nodes):
        #        zp, zq = zip(*zn)
        #        zp   = np.mean(zp, axis=0)
        #        zq   = qmath_np.qmean(zq)
        #        zx   = np.concatenate([zp,zq], axis=0)
        #        nodes.append(zx)

        #    self._slam.optimize(nodes, edges, n_iter=100)
        #    lms = nodes[-self._num_markers:]
        #    lms = [qmath_np.x2pq(x) for x in lms]
        #    self._pub_ze.publish(msgn(lms,rospy.Time.now())) 
        #    #print nodes[-self._num_markers:]
        #    self._graph.clear()

        dp, dq = qmath_np.xrel(self._p, self._q, p, q)
        dx     = np.concatenate([dp,dq], axis=0)

        # save p-q
        #self._p = p
        #self._q = q

        # form dz ...
        mu = self._slam.step(x=dx, zs=zs)
        mu = np.reshape(mu, [-1, 7])
        ep, eq = qmath_np.x2pq(mu[1])
        gtp, gtq = fk(self._dh, j)

        perr = np.subtract(ep,  gtp)
        perr = np.linalg.norm(perr)
        qerr = qmath_np.T(qmath_np.qmul(qmath_np.qinv(gtq), eq))
        qerr = ((qerr + np.pi) % (2*np.pi)) - np.pi
        qerr = np.linalg.norm(qerr)
        epe, eqe = perr, qerr

        perr = np.subtract(p,  gtp)
        perr = np.linalg.norm(perr)
        qerr = qmath_np.T(qmath_np.qmul(qmath_np.qinv(gtq), q))
        qerr = ((qerr + np.pi) % (2*np.pi)) - np.pi
        qerr = np.linalg.norm(qerr)
        pe, qe = perr, qerr
        #print pe-epe, qe-eqe

        ez     = [qmath_np.x2pq(e) for e in mu[2:]]
        #ez     = [unparametrize(e) for e in mu[1:]]

        self._p, self._q = ep.copy(), eq.copy()
        #self._p, self._q = p, q

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
