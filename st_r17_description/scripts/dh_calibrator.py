import numpy as np
import sympy
import tensorflow as tf

def get_dh(index, alpha0, a0, d0):
    with tf.variable_scope('DH_{}'.format(index)):
        alpha = tf.Variable(dtype=tf.float32, initial_value=alpha0, trainable=True, name='alpha'.format(index))
        a = tf.Variable(dtype=tf.float32, initial_value=a0, trainable=True, name='a'.format(index))
        d = tf.Variable(dtype=tf.float32, initial_value=d0, trainable=True, name='d'.format(index))
        q = tf.placeholder(dtype=tf.float32, shape=[None], name='q'.format(index))
    return (alpha, a, d, q)

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
    # full transformation
    dh0 = get_dh(0, 0, 0, 0) # link_0 -> link_1
    dh1 = get_dh(1, 0, 0, 0) # link_1 -> link_2
    dh2 = get_dh(2, 0, 0, 0) # link_2 -> link_3
    dh3 = get_dh(3, 0, 0, 0) # link_3 -> link_4
    dh4 = get_dh(4, 0, 0, 0) # link_4 -> link_5
    T_f = tf.placeholder(dtype=tf.float32, shape=[None,4,4], name='T_f') # link_5 -> object

    dhs = [dh0,dh1,dh2,dh3,dh4]
    Ts = [dh2T(*dh) for dh in dhs]
    Ts.append(T_f)
    T = reduce(lambda a,b : tf.matmul(a,b), Ts) # base_link -> object
    return T

def main():
    T = build()
    loss = tf.square(T - tf.reduce_mean(T, axis=0, keep_dims=True))
    loss = tf.reduce_mean(loss)
    train = tf.train.GradientDescentOptimizer(learning_rate=1e-3).minimize(loss)

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
