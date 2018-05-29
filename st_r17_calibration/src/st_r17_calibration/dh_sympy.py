import numpy as np

import sympy as sm
from sympy import Matrix, MatrixSymbol

eps = np.finfo(float).eps * 4.0
def T2xyzrpy(T):
    # assume T = (N,M,4,4) or (N,4,4)
    i,j,k = (0,1,2)

    x = T[0,3]
    y = T[1,3]
    z = T[2,3]

    cy = sm.sqrt(T[i,i]*T[i,i] + T[j,i]*T[j,i])

    eps_mask = (cy > eps)

    ax0 = sm.atan2(T[k,j],  T[k,k])
    ay = sm.atan2(-T[k,i],  cy)
    az0 = sm.atan2( T[j,i],  T[i,i])

    ax1 = sm.atan2(-T[j,k],  T[j,j])
    az1 = 0.0#sm.zeros(az0.shape)

    ax = eps_mask * ax0 + (1.0-eps_mask) * ax1 
    az = eps_mask * az0 + (1.0-eps_mask) * az1 

    xyzRPY = [sm.simplify(e) for e in [x,y,z,ax,ay,az]]
    xyzRPY = sm.Matrix(xyzRPY)
    return xyzRPY

""" DH Calibrator """
class DHCalibrator(object):
    def __init__(self, dh0, m):
        self._dh0 = np.float32(dh0)
        self._m = m
        self._initialized = False
        self._build()
        self._log()

    @staticmethod
    def _dh(index):
        """ Seed DH Parameters; note q0 is offset joint value. """
        alpha, a, d, dq, q = sm.symbols('alpha{0}, a{0}, d{0}, dq{0}, q{0}'.format(index))
        return (alpha, a, d, q, dq), q

    @staticmethod
    def _dh2T(alpha, a, d, q, dq):
        """ Convert DH Parameters to Transformation Matrix """
        cq = sm.cos(q + dq)
        sq = sm.sin(q + dq)
        ca = sm.cos(alpha)
        sa = sm.sin(alpha)

        T = Matrix([
            [cq, -sq, 0, a],
            [sq*ca, cq*ca, -sa, -sa*d],
            [sq*sa, cq*sa, ca, ca*d],
            [0, 0, 0, 1]
            ])
        return T

    def _build(self):

        def _simplify(x):
            print 'Yay!'
            return sm.simplify(x)

        n = len(self._dh0)
        dhs, qs = zip(*[self._dh(i) for i in range(n)])

        T07 = MatrixSymbol('T07', 4, 4) # input measurement base_link -> object
        T67 = sm.MatrixSymbol('T67', 4, 4) # input measurement stereo_link -> object

        Ts = [self._dh2T(*dh) for dh in dhs]
        Ts.append(T67)

        T = reduce(lambda a, b : _simplify(a*b), Ts)

        psi = xyzRPY_p.jacobian(dhs)
        xyzRPY_p = T2xyzrpy(T) # prediction
        xyzRPY_t = T2xyzrpy(T07) # target

        dr = xyzRPY_t - xyzRPY_p
        d_dh = psi.pinv() * dr

        self._T07 = T07
        self._T67 = T67
        self._dhs = dhs
        self._qs = qs
        self._Ts = Ts
        self._T = T
        self._psi = psi
        self._xyzRPY = xyzRPY_p
        self._dr = dr
        self._d_dh = d_dh

    def _log(self):
        # evaluate dr at current ?
        pass

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

if __name__ == "__main__":
    dh0 = [[3.141592653589793, 0.0, -0.355, 0.0],
            [1.5707963267948966, 0.0, 0.0, -1.5707963267948966],
            [0.0, 0.375, 0.0, 0.0],
            [0.0, 0.375, 0.024, 1.5707963267948966],
            [1.5707963267948966, 0.0, 0.042, 1.176],
            [0.0, -0.012, 0.159, 3.141592653589793]]
    m = 4
    calib = DHCalibrator(dh0, m)

