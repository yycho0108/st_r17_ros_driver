import qmath
import numpy as np
from collections import defaultdict

def dh2T(dh,q):
    alpha, a, d, dq = dh
    cq = np.cos(q + dq)
    sq = np.sin(q + dq)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    res = [[cq,-sq,0,a],
           [sq*ca, cq*ca, -sa, -sa*d],
           [sq*sa,cq*sa,ca,ca*d],
           [0,0,0,1]]
    res = np.asarray(res, dtype=np.float64)
    return res

def dRdDH(dh):
    """ d(R_i) / d(dh_i) """
    alpha,a,d,q = dh

    cq = np.cos(q)
    sq = np.sin(q)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    res = [[[0,0,0,-sq],[0,0,0,-cq],[0,0,0,0]],
         [[-sq*sa,0,0,cq*ca],[-cq*sa,0,0,-ca*sq],[-ca,0,0,0]],
         [[ca*sq,0,0,cq*sa],[cq*ca,0,0,-sq*sa],[-sa,0,0,0]]]
    return np.asarray(res, dtype=np.float64)

def dtdDH(dh):
    """ d(t_i) / d(dh_i) """
    alpha,a,d,q = dh
    cq = np.cos(q)
    sq = np.sin(q)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    res = [[0,1,0,0],[-ca*d,0,-sa,0],[-sa*d,0,ca,0]]
    return np.asarray(res, dtype=np.float64)

def dqdDH(dh):
    """ d(q_i) / d(dh_i) """
    alpha,a,d,q = dh

    sa2 = np.sin(alpha/2)
    ca2 = np.cos(alpha/2)
    sq2 = np.sin(q/2)
    cq2 = np.cos(q/2)

    res = [[ ca2*cq2, 0, 0, -sa2*sq2],
           [-ca2*sq2, 0, 0, -cq2*sa2],
           [-sa2*sq2, 0, 0,  ca2*cq2],
           [-cq2*sa2, 0, 0, -ca2*sq2]]
    return np.asarray(res)

def dot(*args):
    return reduce(lambda a,b : a.dot(b), args)

class RTree(object):
    def __init__(self, Rs, copy=True):
        self._Rs = np.copy(Rs) if copy else Rs
        self._tree = {}
        self._build()
    def _build(self):
        n = len(self._Rs)
        R0 = np.eye(3,3)
        self._tree[0,0] = R0
        for i, R in enumerate(self._Rs):
            R0 = np.dot(R0,R)
            self._tree[0, i+1] = R0
    def __getitem__(self, i0i1):
        i0, i1 = i0i1
        return self._tree[0,i0].T.dot(self._tree[0,i1])

def testDelta(ra, rc, tt):
    res = [[[[ra[0,0]*
        (rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[0,0]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[0,0]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[0,1]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[0,1]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[0,1]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[0,2]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[0,2]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[0,2]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])]]],
    [[[ra[1,0]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[1,0]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[1,0]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[1,1]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[1,1]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[1,1]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[1,2]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[1,2]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[1,2]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])]]],
    [[[ra[2,0]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[2,0]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[2,0]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[2,1]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[2,1]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[2,1]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])],
      [ra[2,2]*(rc[0,0]*tt[0] + rc[0,1]*tt[1] + rc[0,2]*tt[2]),
       ra[2,2]*(rc[1,0]*tt[0] + rc[1,1]*tt[1] + rc[1,2]*tt[2]),
       ra[2,2]*(rc[2,0]*tt[0] + rc[2,1]*tt[1] + rc[2,2]*tt[2])]]]]
    res = np.asarray(res, dtype=np.float64)
    return res

def dedDH(dhs, qs, t0n, q0n):
    """
    dh = dh_i = (\{alpha},a,d,q)
    Rs = (R_0^1, R_1^2, ... R_{n-1}^{n})
    """
    # TODO : remove lots of duplicate calculations

    # initialization ...
    n = len(dhs)
    Ts = [dh2T(dh,q) for (dh,q) in zip(dhs,qs)]
    Rs = [T[:3,:3] for T in Ts]
    ts = [T[:3,3]  for T in Ts]

    # compute prediction
    Tf = dot(*Ts)
    t  = Tf[:3,3]
    q  = qmath.R2q(Tf[:3,:3])

    # compute error
    t_err = t - t0n
    q_err = qmath.qmul(qmath.qinv(q0n), q)

    rtree = RTree(Rs)
    
    dtdx = qmath.dTdX(q_err)
    Q0n  = qmath.ql2Q(q0n)

    res = []
    for (i,dh) in enumerate(dhs):
        deddh_i = np.zeros(shape=(6,4), dtype=np.float64)
        dtddh = dtdDH(dh)
        drddh = dRdDH(dh)
        R_pre = rtree[0,i]
        deddh_i[:3,:] = R_pre.dot(dtddh)

        #dts = np.zeros(shape=(1,3), dtype=np.float64)
        #lhs = np.zeros(shape=(3,1,3,3), dtype=np.float64)
        #for j in range(i+1, n):
        #    R_post = rtree[i+1,j]
        #    lhs += testDelta(R_pre, R_post, ts[j])
        #    dts += R_post.dot(ts[j]).T
        #lhs = np.squeeze(lhs,axis=1)
        #R_pre_s = np.reshape(R_pre, [3,1,3])
        #dts = np.reshape(dts, [1,3,1])
        # TODO : check if correct
        #deddh_i[:3,:] += np.einsum('abc,abd->cd', lhs, drddh)
        #deddh_i[:3,:] += np.einsum('abc,acd->bd', lhs, drddh)
        #deddh_i[:3,:] += np.einsum('abc,bad->cd', lhs, drddh)
        #deddh_i[:3,:] += np.einsum('abc,bcd->ad', lhs, drddh)
        #deddh_i[:3,:] += np.einsum('abc,cad->bd', lhs, drddh)
        #deddh_i[:3,:] += np.einsum('abc,cbd->ad', lhs, drddh)
        q_pre  = qmath.R2q(rtree[0,i])
        q_post = qmath.R2q(rtree[i+1,n])
        #deddh_i[3:,:] = dot(dtdx, Q0n,
        #        qmath.ql2Q(q_pre), # 0 ~ i-1
        #        qmath.qr2Q(q_post), # i+1 ~ n
        #        dqdDH(dh)
        #        )
        #== d(err)/d(dh_i)
        res.append(deddh_i)
    res = np.stack(res, axis=0) # [N_JNT, N_ERR, N_DHP]
    return t_err, q_err, res

def get_delta(dh0, dh1):
    delta = np.subtract(dh1, dh0)
    # normalize + print error
    delta[:,0] = (delta[:,0] + np.pi) % (2*np.pi) - np.pi
    delta[:,3] = (delta[:,3] + np.pi) % (2*np.pi) - np.pi
    return np.sum(np.square(delta))

class DHCalibratorNLLS(object):
    """ Nonlinear Least-Squares DH Calibration """
    def __init__(self):
        pass

def main():
    N_EPOCH = 200
    N_STEPS = 1

    #np.random.seed(0)
    dhs = [[3.141592653589793, 0.0, -0.355, 0.0],
          [1.5707963267948966, 0.0, 0.0, -1.5707963267948966],
          [0.0, 0.375, 0.0, 0.0],
          [0.0, 0.375, 0.024, 1.5707963267948966],
          [1.5707963267948966, 0.0, 0.042, 1.176],
          [0.0, -0.012, 0.159, 3.141592653589793]]
    dhs = np.asarray(dhs, dtype=np.float64)

    print 'ground truth'
    print dhs
    print '============'

    dh0 = np.random.normal(loc=dhs, scale=0.01)
    print 'initial'
    print dh0
    print '---'
    print get_delta(dhs, dh0)
    print '======='

    for _ in range(N_EPOCH):
        ddh = np.zeros_like(dh0)
        for _ in range(N_STEPS):
            #print get_delta,dhs, dh0)
            # joint values
            qs = np.random.uniform(-np.pi, np.pi, size=6)

            # ground truth
            Ts = [dh2T(dh,q) for (dh,q) in zip(dhs,qs)]
            Tf = dot(*Ts)
            t0n = Tf[:3,3]
            q0n = qmath.R2q(Tf[:3,:3])

            # noisy measurements
            dt = qmath.rt(0.0)
            dq = qmath.rq(0.0)
            t0n_z = t0n + dt
            q0n_z = qmath.qmul(dq,q0n)
            qs_z  = np.random.normal(loc=qs, scale=0.0)

            # compute prediction
            t_err, q_err, deddh = dedDH(dh0, qs_z, t0n_z, q0n_z)
            err = np.concatenate([t_err, qmath.T(q_err)], axis=0)
            #err   = [N_ERR]

            # ORDER of deddh = [N_JNT, N_ERR, N_DHP]
            deddh = np.swapaxes(deddh, 1, 0) #reorder: [N_ERR, N_JNT, N_DHP]
            deddh = np.reshape(deddh, [-1, np.size(dh0)]) # [N_ERR, N_JNT*N_DHP]
            ddh  += np.reshape(
                    np.linalg.lstsq(deddh, -err, rcond=None)[0],
                    dh0.shape)
        dh0 = dh0 + np.reshape(ddh / N_STEPS, dh0.shape)
        dh0[:,0] = (dh0[:,0] + np.pi) % (2*np.pi) - np.pi
        dh0[:,3] = (dh0[:,3] + np.pi) % (2*np.pi) - np.pi
        print get_delta(dhs, dh0)

    #dh0 = dh0 + np.reshape(ddh / 1000.0, dh0.shape)

    print 'final'
    print dh0
    print '---'
    print get_delta(dhs, dh0)
    print '====='

if __name__ == "__main__":
    main()
