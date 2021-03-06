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
from st_r17_calibration import qmath
from st_r17_calibration.srv import SetDH, SetDHRequest, SetDHResponse
from st_r17_calibration.msg import DH

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
        self._batch_size  = rospy.get_param('~batch_size', default=64)
        self._noise = rospy.get_param('~noise', default=0.01)
        self._slop = rospy.get_param('~slop', default=0.01)
        self._marquardt = rospy.get_param('~marquardt', default=0.01)

        self._dh = rospy.get_param('~dh', default=None)
        if self._dh is None:
            raise ValueError('Improper DH Parameters Input : {}'.format(self._dh))
        self._dh = np.float32(self._dh)

        # add noise to dh ... or else.
        if self._noise > 0.0:
            # for testing with virtual markers, etc.
            z = np.random.normal(
                    loc = 0.0,
                    scale=self._noise,
                    size = np.shape(self._dh)
                    )
            z = np.clip(z, -2*self._noise, 2*self._noise)
            dh = np.add(self._dh, z)
        else:
            # for testing for real!
            dh = self._dh.copy()
            
        self._m2i = defaultdict(lambda : len(self._m2i))
        self._i2m = {}
        self._dh0 = dh
        self._initialized = False
        self._graph = deque(maxlen=1000)
        self._zinit = False
        self._z_nodes = [[] for _ in range(self._num_markers)]
        self._slam = GraphSlam3(
                n_fix=1+self._num_markers,
                n_dyn=self._batch_size,
                lev=self._marquardt)

        # default observation fisher information
        # TODO : configure, currently just a guess

        #cov = np.square([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        #cov = np.divide(1.0, cov)
        #omega = np.diag(cov)
        #self._omega = omega
        omega = np.diag([100,100,100,100,100,100]).astype(np.float32)
        self._omega = omega
        #self._omega = np.eye(6)

        self._pub_ee = rospy.Publisher('end_effector_est', PoseStamped, queue_size=10)
        self._pub_ze = rospy.Publisher('base_to_target_viz', PoseArray, queue_size=10)
        self._gt_pub = rospy.Publisher('/base_to_target', AprilTagDetectionArray, queue_size=10)
        self._dh_sub = rospy.Subscriber('/dh_params', DH, self.dh_cb)

        self._sub = ApproximateSynchronizer(slop=self._slop, fs=[
            message_filters.Subscriber('joint_states', JointState), # ~50hz, 0.02
            message_filters.Subscriber('stereo_to_target', AprilTagDetectionArray) # ~10hz, 0.1
            ], queue_size=20)

        self._sub.registerCallback(self.data_cb)

    def dh_cb(self, req):
        try:
            dh = req.data
            dh = np.reshape(dh, [-1,4])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'Setting DH Failed : {}'.format(dh))
        #self._dh0 = dh

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

        # testing ; add motion noise
        # p = np.random.normal(p, scale=0.01)
        # dq = qmath.rq(s=0.01)
        # q = qmath.qmul(dq, q)

        if not self._initialized:
            # save previous pose and don't proceed
            self._p = p
            self._q = q
            #self._slam.initialize(np.concatenate([p,q], axis=0))
            self._initialized = True
            return

        # form observations ...
        # nodes organized as:
        # [base_link, landmarks{0..m-1}, poses{0..p-1}]

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

                # testing; add observation noise
                # zp = np.random.normal(zp, scale=0.05)
                # dzq = qmath.rq(s=0.05)
                # zq = qmath.qmul(dzq, zq)

                dz = np.concatenate([zp,zq], axis=0)
                zi = self._m2i[m_id]
                if zi >= self._num_markers:
                    continue
                self._i2m[zi] = m_id
                zs.append([zi, dz, self._omega.copy()])
        except Exception as e:
            rospy.logerr_throttle(1.0, 'TF Failed : {}'.format(e))
            return

        ### semi-offline batch optimization
        ox = 1.0 * self._omega
        self._graph.append([p, q, zs])
        if len(self._graph) >= self._batch_size:
            n_poses = 1 + len(self._graph) # add 1 for 0-node constraint
            nodes = []
            edges = []
            p_nodes = []
            z_nodes = [[] for _ in range(self._num_markers)]

            # add 0-node
            nodes.append( np.asarray([0,0,0, 0,0,0,1]) )

            for gi, g in enumerate(self._graph):
                xi = 1 + self._num_markers + gi # current pose index

                # pose info
                gp, gq, gz = g
                gx = np.concatenate([gp,gq], axis=0)
                p_nodes.append(gx)

                edges.append([0, xi, gx, ox])
                # TODO : handle scenarios when a landmark didn't appear in 200 observations

                for zi, dz, zo in gz:
                    # edge from motion index to re-computed landmark index
                    edges.append([xi,1+zi,dz,zo])
                    zp_a, zq_a = qmath.x2pq(qmath.xadd_rel(gx, dz, T=False))
                    z_nodes[zi].append([zp_a, zq_a])

            # add landmark nodes
            if self._zinit:
                # use the previous initialization for landmarks
                for zp, zq in self._z_nodes:
                    zx = np.concatenate([zp,zq], axis=0)
                    nodes.append(zx)
            else:
                #initialize landmarks with average over 200 samples
                for zi, zn in enumerate(z_nodes):
                    zp, zq = zip(*zn)
                    zp   = np.mean(zp, axis=0)
                    zq   = qmath.qmean(zq)
                    zx   = np.concatenate([zp,zq], axis=0)
                    nodes.append(zx)
                lms = nodes[1:1+self._num_markers]
                lms = [qmath.x2pq(x) for x in lms]
                lms_re = [lms[self._m2i[i]] for i in range(self._num_markers)]
                self._pub_ze.publish(msgn(lms_re,rospy.Time.now())) 
                self._zinit = True

            # add motion nodes
            for zx in p_nodes:
                nodes.append(zx)

            #print "PRE:"
            #print nodes[-self._num_markers:]
            nodes = self._slam.optimize(nodes, edges, max_iter=100, tol=1e-12, min_iter=100)
            #print "POST:"
            #print nodes[-self._num_markers:]

            lms = nodes[1:1+self._num_markers] # landmarks
            lms = [qmath.x2pq(x) for x in lms]
            self._z_nodes = lms
            lms_re = [lms[self._m2i[i]] for i in range(self._num_markers)]
            self._pub_ze.publish(msgn(lms_re,rospy.Time.now())) 
            self._graph.clear()

            stamp = rospy.Time.now()
            g_msg = AprilTagDetectionArray()
            g_msg.header.stamp = stamp
            g_msg.header.frame_id = 'base_link'

            for i in range(self._num_markers):
                msg = AprilTagDetection()
                m_id = self._i2m[i]
                msg.id = [m_id]
                msg.size = [0.0] # not really a thing
                msg.pose.pose.pose = pose(lms[i][0], lms[i][1])
                msg.pose.header.frame_id = 'base_link'
                msg.pose.header.stamp = stamp
                g_msg.detections.append(msg)
            self._gt_pub.publish(g_msg)
        return
        ### END: semi-offline batch optimization

        ### online slam with "reasonable" initialization?
        #if not self._zinit:
        #    # add entry ...
        #    gx = np.concatenate([p,q], axis=0)
        #    for z in zs:
        #        _, zi, dz, _ = z
        #        zi = zi - 2
        #        zp_a, zq_a = qmath.x2pq(qmath.xadd_rel(gx, dz, T=False))
        #        self._z_nodes[zi].append( (zp_a, zq_a) )

        #    # check entries ...
        #    ls = [len(e) for e in self._z_nodes]
        #    if np.all(np.greater_equal(ls, 100)):
        #        znodes = []
        #        for zi, zn in enumerate(self._z_nodes):
        #            zp, zq = zip(*zn)
        #            zp   = np.mean(zp, axis=0)
        #            zq   = qmath.qmean(zq)
        #            zx   = np.concatenate([zp,zq], axis=0)
        #            znodes.append(zx)
        #            self._slam._nodes[2+zi] = zx
        #            self._slam.initialize(gx)
        #            self._p = p
        #            self._q = q
        #        self._zinit = True
        #    return
        ### END : online slam with reasonable initialization

        #dp, dq = qmath.xrel(self._p, self._q, p, q)
        #dx     = np.concatenate([dp,dq], axis=0)
        dx      = np.concatenate([p,q], axis=0)
        self._slam._nodes[0] = np.asarray([0,0,0,0,0,0,1], dtype=np.float32)

        # save p-q
        #self._p = p
        #self._q = q

        #        for zp, zq in self._z_nodes:
        #            zx = np.concatenate([zp,zq], axis=0)
        #            nodes.append(zx)

        # form dz ...
        mu = self._slam.step(x=dx, zs=zs)
        mu = np.reshape(mu, [-1, 7])
        ep, eq = qmath.x2pq(mu[1])
        gtp, gtq = fk(self._dh, j)

        #perr = np.subtract(ep,  gtp)
        #perr = np.linalg.norm(perr)
        #qerr = qmath.T(qmath.qmul(qmath.qinv(gtq), eq))
        #qerr = ((qerr + np.pi) % (2*np.pi)) - np.pi
        #qerr = np.linalg.norm(qerr)
        #epe, eqe = perr, qerr

        #perr = np.subtract(p,  gtp)
        #perr = np.linalg.norm(perr)
        #qerr = qmath.T(qmath.qmul(qmath.qinv(gtq), q))
        #qerr = ((qerr + np.pi) % (2*np.pi)) - np.pi
        #qerr = np.linalg.norm(qerr)
        #pe, qe = perr, qerr
        #print pe-epe, qe-eqe

        ez     = [qmath.x2pq(e) for e in mu[2:]]

        stamp = rospy.Time.now()
        g_msg = AprilTagDetectionArray()
        g_msg.header.stamp = stamp
        g_msg.header.frame_id = 'base_link'

        for i in range(self._num_markers):
            msg = AprilTagDetection()
            m_id = self._i2m[i]
            msg.id = [m_id]
            msg.size = [0.0] # not really a thing
            msg.pose.pose.pose = pose(ez[i][0], ez[i][1])
            msg.pose.header.frame_id = 'base_link'
            msg.pose.header.stamp = stamp
            g_msg.detections.append(msg)
        self._gt_pub.publish(g_msg)

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
