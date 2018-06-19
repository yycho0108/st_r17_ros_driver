"""
GraphSlam 3D Implementation.
"""

import numpy as np
import qmath
import scipy.linalg

def block(ar):
    """ Convert Block Matrix to Dense Matrix """
    ni,nj,nk,nl = ar.shape
    return np.swapaxes(ar, 1, 2).reshape(ni*nk, nj*nl)

def unblock(ar, nknl):
    """ Convert Dense Matrix to Block Matrix """
    nk,nl = nknl
    nink, njnl = ar.shape
    ni = nink/nk
    nj = njnl/nl
    return ar.reshape(ni,nk,nj,nl).swapaxes(1,2)

class GraphSlam3(object):
    def __init__(self, n_l, l=0.0):
        self._nodes = {}
        self._n_l = n_l
        self._lambda = l

    def add_node(self, i,
            z, o_z, x=None, o_x=None):
        """
        i = node index
        z = observation
        o_z = covariance of observation
        x = source
        o_x = covariance of source

        TODO : incomplete
        """
        self._nodes[i] = z
        #zp,zq = qmath.x2pq(z)
        #xp,xq = qmath.x2pq(x)

        ## compute GR
        #GR0 = np.hstack([np.eye(3), qmath.q2R(xq).T.dot(zp)])
        #GR1 = qmath.qr2Q(zq)
        #GR = np.vstack([GR0,GR1])

        ## compute Gy
        #GY0 = q2R(xq)
        #GY1 = ql2Q(xq)
        #GY = np.vstack([GY0,GY1])

        #P_LL = GR.dot(o_x).dot(GR.T) + GY.dot(o_z).dot(GY.T)
        #P_LX = GR.dot(P_RM)

    def del_node(self, i):
        pass

    def add_edge(self, x, i0, i1, nodes=None):
        if nodes is None:
            n = self._nodes
        else:
            n = nodes

        p0, q0 = qmath.x2pq(n[i0])
        p1, q1 = qmath.x2pq(n[i1])
        dp, dq = qmath.x2pq(x)
        Aij, Bij, eij = qmath.Aij_Bij_eij(p0,p1,dp,q0,q1,dq)
        return Aij, Bij, eij

    def initialize(self, x0):
        n = 2 + self._n_l # [x0,x1,l0...ln]
        self._H = np.zeros((n,n,6,6), dtype=np.float64)
        self._b = np.zeros((n,1,6,1), dtype=np.float64)
        #self._H[0,0] = 1000*np.eye(6) << what would this do?
        self._nodes[0] = x0
        x,q = qmath.x2pq(x0)

        #print 'x', x
        #print 'q', q, np.linalg.norm(q)
        #raise "Stop"

        # TODO : Are the below initializations necessary?
        #p, q = qmath.x2pq(x0)
        #x = np.concatenate([p,qmath.T(q)], axis=-1)
        #self._b[0,0,:,0] = x

    def optimize(self, nodes, edges, n_iter=10, tol=1e-4, min_iter=0):
        """ Offline Version """
        n = len(nodes)
        for it in range(n_iter):
            H = np.zeros((n,n,6,6), dtype=np.float64)
            b = np.zeros((n,1,6,1), dtype=np.float64)
            for z0, z1, z, o in edges:
                Aij,Bij,eij = self.add_edge(z, z0, z1, nodes)
                H[z0,z0] += Aij.T.dot(o).dot(Aij)
                H[z0,z1] += Aij.T.dot(o).dot(Bij)
                H[z1,z0] += Bij.T.dot(o).dot(Aij)
                H[z1,z1] += Bij.T.dot(o).dot(Bij)
                b[z0]   += Aij.T.dot(o).dot(eij)
                b[z1]   += Bij.T.dot(o).dot(eij)

            # fix node 0
            H[0,0] += 1e9 * np.eye(6,6)

            H = block(H)
            b = block(b)

            #mI = self._lambda * np.eye(*H.shape) # marquardt damping
            mI = np.diag(np.diag(H))* self._lambda
            #dx = np.linalg.lstsq(H+mI,-b, rcond=None)[0]
            dx = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(H+mI), -b)
            dx = np.reshape(dx, [-1,6]) # [x1, l0, ... ln]

            for i in range(n):
                nodes[i] = qmath.xadd_abs(nodes[i], dx[i])

            delta = np.sum(np.square(dx))
            if delta < tol:
                print 'it : {}'.format(it)
                #print 'solved'
                break
        return nodes

    def step(self, x=None, zs=None):
        """ Online Version """

        # " expand "
        self._H[1,:] = 0.0
        self._H[:,1] = 0.0
        self._b[1]   = 0.0

        zis = [] # updates list

        ox = np.diag([1,1,1,1,1,1])
        # apply motion updates first
        if x is not None:
            self._nodes[1] = qmath.xadd_rel(self._nodes[0], x, T=False)
            zis.append(1)
            #zs.append([0,1,x,ox])
            #zs = zs + [[0, 1, x, ox]]

        # H and b are organized as (X0, X1, L0, L1, ...)
        # where X0 is the previous position, and X1 is the current position.
        # Such that H[0,..] pertains to X0, and so on.

        # now with observations ...
        for (z0, z1, z, o) in zs:
            zis.append(z1)
            if z1 not in self._nodes:
                # initial guess
                self._nodes[z1] = qmath.xadd_rel(
                        self._nodes[z0], z, T=False)
                # no need to compute deltas for initial guesses
                # (will be zero) ???
                # continue
            Aij, Bij, eij = self.add_edge(z, z0, z1)
            self._H[z0,z0] += Aij.T.dot(o).dot(Aij)
            self._H[z0,z1] += Aij.T.dot(o).dot(Bij)
            self._H[z1,z0] += Bij.T.dot(o).dot(Aij)
            self._H[z1,z1] += Bij.T.dot(o).dot(Bij)
            self._b[z0]   += Aij.T.dot(o).dot(eij)
            self._b[z1]   += Bij.T.dot(o).dot(eij)

        H00 = block(self._H[:1,:1])
        H01 = block(self._H[:1,1:])
        H10 = block(self._H[1:,:1])
        H11 = block(self._H[1:,1:])

        B00 = block(self._b[:1,:1])
        B10 = block(self._b[1:,:1])

        AtBi = np.matmul(H10, np.linalg.pinv(H00))
        XiP  = B10

        # fold previous information into new matrix
        H = H11 - np.matmul(AtBi, H01)
        B = B10 - np.matmul(AtBi, B00)

        mI = self._lambda * np.eye(*H.shape) # marquardt damping
        #dx = np.matmul(np.linalg.pinv(H+mI), -B)
        dx = np.linalg.lstsq(H+mI,-B, rcond=None)[0]
        dx = np.reshape(dx, [-1,6]) # [x1, l0, ... ln]
        delta = np.sum(np.abs(dx))
        #print 'delta', delta

        #if delta < 1.0:
        #for i in zis:
        #    self._nodes[i] = qmath.xadd_abs(self._nodes[i], dx[i-1])
        try:
            self._alpha *= 1.0#0.9999
        except Exception:
            self._alpha = 1.0
        for i in range(1, 2+self._n_l):
            if i in self._nodes:
                self._nodes[i] = qmath.xadd_abs(self._nodes[i], self._alpha*dx[i-1])

        # replace previous node with current position
        #self._nodes[0] = self._nodes[1].copy()

        H = unblock(H, (6,6))
        B = unblock(B, (6,1))

        # assign at appropriate places, with x_0 being updated with x_1
        self._H[:1,:1] = H[:1,:1]
        self._H[:1,2:] = H[:1,1:]
        self._H[2:,:1] = H[1:,:1]
        self._H[2:,2:] = H[1:,1:]
        self._b[:1] = B[:1]
        self._b[2:] = B[1:]

        x = [self._nodes[k] for k in sorted(self._nodes.keys())]
        return x
