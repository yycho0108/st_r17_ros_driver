#!/usr/bin/python2

import numpy as np
import tensorflow as tf

eps=np.finfo(float).eps*16.0

""" Utility """
def T2xyzrpy(T):
    with tf.name_scope('T2xyzrpy', [T]):
        # assume T = (N,M,4,4) or (N,4,4)
        i,j,k = (0,1,2)

        x = T[...,0,3]
        y = T[...,1,3]
        z = T[...,2,3]

        cy = tf.sqrt(
                tf.square(T[...,i,i]) + 
                tf.square(T[...,j,i])
                )

        eps_mask = tf.greater(cy, eps)#, tf.float32)

        ax0 = tf.atan2( T[...,k,j],  T[...,k,k])
        ay = tf.atan2(-T[...,k,i],  cy)
        az0 = tf.atan2( T[...,j,i],  T[...,i,i])

        ax1 = tf.atan2(-T[...,j,k],  T[...,j,j])
        az1 = tf.zeros_like(az0)

        ax = tf.where(eps_mask, ax0, ax1)# * ax0 + (1.0 - eps_mask) * ax1 
        az = tf.where(eps_mask, az0, az1)#eps_mask * az0 + (1.0 - eps_mask) * az1 

        #xyz = tf.stack([x,y,z], axis=-1)
        #rpy = tf.stack([ax,ay,az], axis=-1)
        xyzRPY = tf.stack([x,y,z,ax,ay,az], axis=-1)
    return xyzRPY

def T2xyzrpy_v2(T):
    T = tf.unstack(T, axis=-2) #
    return tf.concat(T[:3], axis=-1)

def qinv(q):
    """ assume unit quaternion """
    qx,qy,qz,qw = tf.unstack(q, axis=-1)
    return tf.stack([-qx,-qy,-qz,qw], axis=-1)

def qmul(qa, qb):
    """ assume unit quaterion """
    va, ra = tf.split(qa, [3, 1], axis=-1)
    vb, rb = tf.split(qb, [3, 1], axis=-1)
    vb = tf.ones_like(va) * vb
    rc = ra*rb - tf.reduce_sum(va*vb, axis=-1, keepdims=True)
    vc = ra*vb + rb*va + tf.cross(va,vb)
    q = tf.concat((vc,rc), axis=-1)
    q = q / tf.norm(q, keepdims=True)
    return q

qxq=qmul # alias

def qlog(q):
    """ assume unit quaternion """
    va, ra = tf.split(q, [3,1], axis=-1)
    uv = va / tf.norm(va, axis=-1, keepdims=True)
    return uv * tf.acos(ra)

def check_nan(t, msg=''):
    c = tf.reduce_any(tf.is_nan(t))
    return tf.Print(t, [c], message=msg)

def T2err(Ta, Tb):
    with tf.name_scope('T2err', [Ta,Tb]):
        a_xyz = Ta[...,:3,3]
        b_xyz = Tb[...,:3,3]

        a_q = T2q(Ta)
        b_q = T2q(Tb)

        dxyz = a_xyz - b_xyz

        b_qi = qinv(b_q)
        qm = qmul(a_q, b_qi)
        dq = qlog(qm)

        #cnb = check_nan(b_qi, 'b_qi')
        #cnqm = check_nan(qm, 'qm')
        #cndq = check_nan(dq, 'dq')

        #na = tf.reduce_any(tf.is_nan(Ta))
        #nb = tf.reduce_any(tf.is_nan(Tb))
        #nq = tf.reduce_any(tf.is_nan(dq))
        #pr = tf.Print(Ta, [na,nb,nq], message='nancheck')
        #with tf.control_dependencies([cnb, cnqm, cndq, pr]):
        err = tf.concat([dxyz,dq], axis=-1) # [...x6]
    return err

def pinv(M):
    MtM = tf.matmul(M, M, transpose_a=True)
    MtMi = tf.matrix_inverse(MtM)
    return tf.matmul(MtMi, M, transpose_b=True)

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

def dh2Tv2(dh, q):
    with tf.name_scope('dh2Tv2', [dh, q]):
        alpha, a, d, dq = tf.unstack(dh, axis=0)
        q = dq + q

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

def T2q(T):
    # accepts (..., 4, 4) homogeneous transformation matrix
    t = tf.trace(T)
    s = tf.unstack(tf.shape(T))
    T_flat = tf.reshape(T, s[:-2]+[16])

    \
            T00,T01,T02,T03,\
            T10,T11,T12,T13,\
            T20,T21,T22,T23,\
            T30,T31,T32,T33 = tf.unstack(T_flat, axis=-1)

    # opt 1 : t > T[..., 3,3]
    q1 = [
            T21-T12,
            T02-T20,
            T10-T01,
            t,
            ]
    q1 = tf.stack(q1, axis=-1)
    q1 = 0.5 * q1 / tf.expand_dims(tf.sqrt(t * T33), -1)

    # opt 2 :
    Td = tf.matrix_diag_part(T)[..., :3] #needed?
    qmx = tf.cast(tf.argmax(Td, axis=-1), tf.int32)

    q_mx0 = tf.stack([
        T00-T11-T22+T33,
        T01+T10,
        T20+T02,
        T21-T12], axis=-1)
    q_mx1 = tf.stack([
        T01+T10,
        T11-T22-T00+T33,
        T12+T21,
        T02-T20], axis=-1)
    q_mx2 = tf.stack([
        T20+T02,
        T12+T21,
        T22-T00-T11+T33,
        T10-T01], axis=-1)

    # is this faster than the following?

    # opt 2.1 : stack + gather
    #qmx3 = tf.stack([q_mx0, q_mx1, q_mx2], axis=0)
    #q2 = tf.gather(qmx3, qmx)

    # opt 2.2 : cast + add
    mx0 = tf.expand_dims(tf.cast(tf.equal(qmx, 0), tf.float32), axis=-1)
    mx1 = tf.expand_dims(tf.cast(tf.equal(qmx, 1), tf.float32), axis=-1)
    mx2 = tf.expand_dims(tf.cast(tf.equal(qmx, 2), tf.float32), axis=-1)
    q2 = (q_mx0*mx0 + q_mx1*mx1 + q_mx2*mx2)

    sel = tf.cast((t>T33)[..., tf.newaxis], tf.float32)
    q = sel*q1 + (1.0-sel)*q2
    q = q / tf.norm(q, axis=-1, keepdims=True)
    return q

def jacobian(y, x):
    n = tf.shape(y)[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, J = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y[j], x))),
        loop_vars)

    return J.stack()

""" DH Calibrator """
class DHCalibrator(object):
    def __init__(self, dh0, m, lr):
        self._dh0 = np.float32(dh0)
        self._m = m
        self._lr = lr
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
        return (alpha, a, d, q+dq), (alpha, a, d, dq), q

    @staticmethod
    def _dhs(dh0):
        """ Seed DH Parameters as a single array"""
        n_j = len(dh0)
        dh = tf.Variable(dtype=tf.float32, initial_value=dh0, trainable=True, name='dh') #(J,4)
        q  = tf.placeholder(dtype=tf.float32, shape=[None, n_j], name='q') #(B,J)
        return dh, q

    def _build(self):
        tf.reset_default_graph()
        self._graph = tf.get_default_graph()
        # inputs ...
        with tf.name_scope('inputs'):
            dhs, qs = self._dhs(self._dh0)
            T_f = tf.placeholder(dtype=tf.float32, shape=[None, self._m, 4,4], name='T_f') # camera_link -> object(s)
            vis = tf.placeholder(dtype=tf.bool, shape=[None, self._m], name='vis') # marker visibility
            T_targ = tf.placeholder(dtype=tf.float32, shape=[self._m, 4, 4], name='T_targ')

        # create aliases ...
        vis_f = tf.cast(vis, tf.float32) # (N, M)

        # build  transformation ...
        with tf.name_scope('transforms'):
            dhs_j = tf.unstack(dhs, axis=0)
            qs_j = tf.unstack(qs, axis=1)
            Ts = [dh2Tv2(dh,q) for (dh,q) in zip(dhs_j,qs_j)]
            T06 = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> stereo_optical_link
            T = tf.einsum('aij,abjk->abik', T06, T_f) # apply landmarks transforms

        #mode = 'xyzrpy'
        mode = 'Gradient'
        #mode = 'Jacobian'

        # T_targ_67 = ...
        # TODO : inverse of homogeneous transform can be found a lot easier.
        # for now, leave in the simplest form.
        
        R = T_f[...,:3,:3]
        d = T_f[...,:3,3:]
        R_td = tf.matmul(R, d, transpose_a=True)
        R_t = tf.matrix_transpose(R)
        T_fi34 = tf.concat([R_t, -R_td], axis=-1)
        T_fi4  = tf.ones_like(T_f[...,:1,:]) * np.reshape([0,0,0,1], [1,1,1,4])
        T_fi44 = tf.concat([T_fi34,  T_fi4], axis=-2)

        #T_targ_06 = tf.einsum('mij,bmjk->bmik', T_targ, tf.matrix_inverse(T_f))
        T_targ_06 = tf.einsum('mij,bmjk->bmik', T_targ, T_fi44)
        #T_targ_06 = (N, M, 4, 4), all of which represent T06

        loss = T2err(T, tf.expand_dims(T_targ, axis=0))
        loss = tf.square(loss)
        loss = tf.reduce_sum(loss * vis_f[..., tf.newaxis]) / (9.0 * tf.reduce_sum(vis_f))

        params = [dhs]

        grad = tf.gradients(loss, params)
        gnan = tf.reduce_any(tf.is_nan(grad))
        #with tf.control_dependencies([gnan]):
        opt = tf.train.AdamOptimizer(learning_rate=self._lr)
        train = tf.cond(gnan, lambda:tf.no_op(), lambda:opt.apply_gradients(zip(grad, params)))
        #train = opt.apply_gradients( zip(grad, params))
            #train = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss)

        #pred_xyzRPY = T2xyzrpy_v2(T06) # == (N, 12)
        #targ_xyzRPY = T2xyzrpy_v2(T_targ_06) # == (N, M, 12)

        #if mode != 'Jacobian':
        #    #loss = tf.square(tf.expand_dims(pred_xyzRPY,1) - targ_xyzRPY) # self position loss, dynamic
        #    loss = tf.square(tf.expand_dims(T_targ,axis=0) - T) # object location loss, static
        #    loss = tf.reduce_sum(loss * vis_f[..., tf.newaxis, tf.newaxis]) / (16.0 * tf.reduce_sum(vis_f))
        #    train = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(loss)
        #else:
        #    targ_xyzRPY = tf.reduce_sum(targ_xyzRPY * vis_f[..., tf.newaxis], axis=1) # (N, 12)
        #    targ_xyzRPY /= tf.reduce_sum(vis_f, axis=1, keep_dims=True) #(N,12)/(N,1)

        #    delta_xyzRPY = targ_xyzRPY - pred_xyzRPY # (N, 12)

        #    #dxyz = delta_xyzRPY[:,:3]
        #    #drpy = delta_xyzRPY[:,3:]
        #    #drpy = tf.atan2(tf.sin(drpy), tf.cos(drpy))
        #    #delta_xyzRPY = tf.concat([dxyz,drpy], axis=-1)

        #    #with tf.control_dependencies([tf.Print(targ_xyzRPY, [tf.reduce_mean(tf.abs(delta_xyzRPY), axis=0)], message='xyzRPY', summarize=6)]):
        #    delta_xyzRPY_1 = tf.reshape(delta_xyzRPY, [-1, 1]) #(Nx12, 1)

        #    # Jacobian Methods
        #    pred_xyzRPY_1 = tf.reshape(pred_xyzRPY, [-1]) # (Nx12,)

        #    psi = jacobian(pred_xyzRPY_1, dhs) #(Nx12, J, 4)
        #    BS,_,J,_ = tf.unstack(tf.shape(psi))
        #    psi = tf.reshape(psi, [BS, J*4])

        #    d_dh = tf.matmul(pinv(psi), delta_xyzRPY_1)
        #    d_dh = tf.reshape(d_dh, dhs.shape) #d_dh = (Jx4) --> (J,4)

        #    #loss = tf.reduce_sum(tf.square(d_dh)) # update magnitude
        #    loss = tf.reduce_sum(tf.square(delta_xyzRPY))
        #    train = tf.assign_add(dhs, 1e-2 * d_dh)

        # Simple Loss
        #loss = tf.reduce_mean(tf.square(delta_xyzRPY))
        #train = tf.train.AdamOptimizer(learning_rate=5e-2).minimize(loss)

        # --> (Jx4, 1)

        #vis_sel = tf.tile(vis[..., tf.newaxis], [1,1,3])

        # save ...
        self._T = T
        self._T_f = T_f
        self._T_targ = T_targ
        self._dhs = dhs
        self._vis = vis
        self._qs = qs_j
        self._loss = loss
        self._train = train

    def _log(self):
        # logs ...
        #tf.summary.scalar('loss_xyz', self._loss_xyz)
        #tf.summary.scalar('loss_rpy', self._loss_rpy)
        tf.summary.scalar('loss', self._loss)

        self._writer = tf.summary.FileWriter('/tmp/dh/13', self._graph)
        self._summary = tf.summary.merge_all()

    def eval_1(self, js, xs, vis):
        feed_dict = {q:[j] for q,j in zip(self._qs, js)}
        feed_dict[self._T_f] = np.expand_dims(xs, 0) # [1, M, 4, 4] 
        feed_dict[self._vis] = np.expand_dims(vis, 0)
        return self.run(self._T, feed_dict = feed_dict)[0]

    def start(self):
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._iter = 0

    def step(self, js, xs, vis, ys):
        feed_dict = {q:j for q,j in zip(self._qs, np.transpose(js))}
        feed_dict[self._T_f] = xs
        feed_dict[self._vis] = vis
        feed_dict[self._T_targ] = ys

        _, loss, dhs, summary = self.run(
                [self._train, self._loss, self._dhs, self._summary],
                feed_dict = feed_dict)
        self._writer.add_summary(summary, self._iter)
        self._iter += 1
        return loss, dhs

    def run(self, *args, **kwargs):
        return self._sess.run(*args, **kwargs)

