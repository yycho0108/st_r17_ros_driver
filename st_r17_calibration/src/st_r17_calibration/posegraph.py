import numpy as np
import qmath as qm
import scipy.linalg

def b2d(ar):
    """ Convert Block Matrix to Dense Matrix """
    ni,nj,nk,nl = ar.shape
    return np.swapaxes(ar, 1, 2).reshape(ni*nk, nj*nl)

def d2b(ar, nknl):
    """ Convert Dense Matrix to Block Matrix """
    nk,nl = nknl
    nink, njnl = ar.shape
    ni = nink/nk
    nj = njnl/nl
    return ar.reshape(ni,nk,nj,nl).swapaxes(1,2)

class PoseGraph(object):
    def __init__(self, ndim=6, size=None, dtype=np.float64):
        self._size = size
        self._ndim = ndim
        self._dtype= dtype

        self._nodes = {}
        self._edges = {}

        if size is not None:
            self._H = np.zeros((size,size,ndim,ndim), dtype=np.float64)
            self._b = np.zeros((size,1,ndim,1), dtype=np.float64)
            self._static = True
        else:
            self._H = None
            self._b = None
            self._static = False

    def add_node(self, node, i0):
        self._nodes[i0] = node

    def add_edge(self, edge, i0, i1):
        self._edges[ (i0,i1) ] = edge

    def del_node(self, i0, i1=None):
        # currently only supports del[i0] or del[i0:i1]
        i1 = (i0+1) if (i1 is None) else (i1)

        # 0 ) load information
        H = self._H
        b = self._b
        
        # 1 ) rearrange array so that H00 is to be marginalized

        # 1.1 ) construct index array forwards/backwards
        idx = np.arange(self._size, dtype=np.int32)
        ix0 = (i0 <= idx) & (idx < i1)
        ix1 = np.logical_not(ix0)
        idx_fw = np.concatenate([idx[ix0], idx[ix1]])
        idx_bw = np.empty(self._size, dtype=np.int32)
        idx_bw[idx_fw] = idx

        # 1.2 ) apply indexed view to H/b matrices
        H = H[idx_fw, :]
        H = H[:, idx_fw]
        b = b[idx_fw]

        # 2 ) extract block submatrices
        n0 = i1 - i0
        H00 = b2d(H[:n0,:n0])
        H01 = b2d(H[:n0,n0:])
        H10 = b2d(H[n0:,:n0])
        H11 = b2d(H[n0:,n0:])
        b00 = b2d(b[:n0,:n0])
        b10 = b2d(b[n0:,:n0])

        # 3) fold previous information into new matrix
        # through schur complement
        H10_H00i = np.matmul(H10, np.linalg.pinv(H00))
        H = H11 - np.matmul(H10_H00i, H01)
        b = b10 - np.matmul(H10_H00i, b00)

        # 4) back to original view
        # TODO : fix here ##################################################
        H = d2b(H, (6,6))
        b = d2b(b, (6,1))
        H = H[idx_bw, :]
        H = H[:, idx_bw]
        b = b[idx_bw]

        # 5) zero out prior information
        H[ix0, :] *= 0
        H[:, ix0] *= 0
        b[ix0]    *= 0

        # 6) save information
        self._H = H
        self._b = b

        # 7) delete node information
        for i in range(i0, i1):
            del self._nodes[i]

    def del_edge(self, i0, i1):
        del self._edges[ (i0, i1) ]

    def reset_edges(self):
        self._edges = {}

    def finalize(self):
        for (i0,i1), (ev,eo) in sorted(self._edges.iteritems()):
            if i1 not in self._nodes:
                self._nodes[i1] = qm.xadd_rel(self._nodes[i0], ev, T=False)
            # TODO: alternatively, take the mean over suggested estimates

        if not self._static:
            size = len(self._nodes)
            self._size = size
            self._H = np.zeros((size,size,ndim,ndim), dtype=np.float64)
            self._b = np.zeros((size,1,ndim,1), dtype=np.float64)

    def optimize(self,
            lev=0.0,
            min_iter=0,
            max_iter=100,
            tol=1e-9):

        self.finalize()

        nodes = self._nodes
        edges = self._edges
        n = len(nodes)

        for it in range(max_iter):
            #H = np.zeros_like(self._H)
            #b = np.zeros_like(self._b)
            H = self._H
            b = self._b

            for (i0,i1), (ev,eo) in self._edges.iteritems():
                # unpack information ...
                p0, q0 = qm.x2pq(self._nodes[i0])
                p1, q1 = qm.x2pq(self._nodes[i1])
                dp, dq = qm.x2pq(ev)

                # compute error ...
                Aij, Bij, eij = qm.Aij_Bij_eij(p0,p1,dp,q0,q1,dq)

                # update ...
                H[i0,i0] += Aij.T.dot(eo).dot(Aij)
                H[i0,i1] += Aij.T.dot(eo).dot(Bij)
                H[i1,i0] += Bij.T.dot(eo).dot(Aij)
                H[i1,i1] += Bij.T.dot(eo).dot(Bij)
                b[i0]   += Aij.T.dot(eo).dot(eij)
                b[i1]   += Bij.T.dot(eo).dot(eij)

            # fix node 0; TODO : valid?
            H[0,0] += 1e3 * np.eye(6,6)
            b[0] = 0

            H = b2d(H)
            b = b2d(b)
            #mI = self._lambda * np.eye(*H.shape) # marquardt damping
            mI = np.diag(np.diag(H)) * lev
            #dx = np.linalg.lstsq(H+mI,-b, rcond=None)[0]
            dx = scipy.linalg.cho_solve(
                    scipy.linalg.cho_factor(H+mI), -b)
            dx = np.reshape(dx, [-1,6]) # [x1, l0, ... ln]

            for i in range(n):
                nodes[i] = qm.xadd_abs(nodes[i], dx[i])

            delta = np.sum(np.square(dx))
            if delta < tol:
                print 'it : {}'.format(it)
                #print 'solved'
                break
        self._H = d2b(H, (6,6))
        self._b = d2b(b, (6,1))
        return nodes


