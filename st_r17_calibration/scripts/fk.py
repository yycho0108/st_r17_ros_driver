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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hide_axis(ax):
    """ Hide Axis (Dummy) """
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

def characterize(txn_err, rxn_err):
    """ Characterize DH Parameter Errors """
    fig = plt.figure(figsize=(9.6, 4.8))

    # dummy axis for setting super title
    ax0 = fig.add_subplot(111)
    ax0.set_title('Kinematic Error Characterization')
    hide_axis(ax0)

    # create axes, Layout [ 1 | 2 ]
    ax_p = fig.add_subplot(121, projection='3d')
    ax_p.set_title('Position Error (m) ')

    ax_p.set_xlabel('x')
    ax_p.set_ylabel('y')
    ax_p.set_zlabel('z')

    ax_q = fig.add_subplot(122, projection='3d')
    ax_q.set_title('Orientation Error (deg) ')

    ax_q.set_xlabel('R')
    ax_q.set_ylabel('P')
    ax_q.set_zlabel('Y')
    

    ax_p.scatter(txn_err[:,0], txn_err[:,1], txn_err[:,2])
    ax_q.scatter(rxn_err[:,0], rxn_err[:,1], rxn_err[:,2])

    plt.show()

def asub(a, b):
    dx = np.subtract(a,b)
    return np.arctan2(np.sin(dx), np.cos(dx))

def main():
    _dh0 = [
            [np.pi, 0, -(0.033 + 0.322), 0],
            [np.pi/2, 0, 0, -np.pi/2],
            [0, 0.375, 0, 0],
            [0, 0.375, 0.024, np.pi/2],
            [np.pi/2, 0, 0.042, 1.176],
            [0, -0.012, 0.159, np.pi]
            ]
    _dh0 = np.float32(_dh0)
    _dh1 = np.loadtxt('/tmp/dhf.csv')
    print _dh0 - _dh1
    #_dh1 = np.concatenate((_dh1, _dh0[:, -1, np.newaxis]), axis=-1)

    d_xyz = []
    d_rpy = []
    for i in range(1000):
        qs = np.random.uniform(-np.pi, np.pi, size=6)
        qs[-1] = 0.0
        txn0, rxn0 = fk(_dh0, qs)
        txn1, rxn1 = fk(_dh1, qs)
        rpy0 = tx.euler_from_quaternion(rxn0)
        rpy1 = tx.euler_from_quaternion(rxn1)
        d_xyz.append( np.subtract(txn0, txn1) )
        d_rpy.append( asub(rpy0, rpy1) )

    d_xyz = np.float32(d_xyz)
    d_rpy = np.rad2deg(np.float32(d_rpy))

    print('d_xyz', np.mean(np.abs(d_xyz), axis=0))
    print('d_rpy', np.mean(np.abs(d_rpy), axis=0))
    characterize(d_xyz, d_rpy)


if __name__ == "__main__":
    main()
