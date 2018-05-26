import numpy as np
import tf.transformations as tx

def dh2T(alpha, a, d, dq, q):
    """ Convert DH Parameters to Transformation Matrix """
    cq = np.cos(q + dq)
    sq = np.sin(q + dq)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.float32([
        [cq, -sq, 0, a],
        [sq*ca, cq*ca, -sa, -sa*d],
        [sq*sa, cq*sa, ca, ca*d],
        [0, 0, 0, 1]
        ])

def fk(dhs, qs):
    Ts = [dh2T(*dh, q=q) for (dh,q) in zip(dhs, qs)]
    T = reduce(lambda a,b : np.matmul(a,b), Ts) # base_link -> object
    txn = tx.translation_from_matrix(T)
    rxn = tx.quaternion_from_matrix(T)
    return txn, rxn


