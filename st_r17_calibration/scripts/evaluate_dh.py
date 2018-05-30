#!/usr/bin/python2

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from st_r17_calibration.kinematics import fk

import numpy as np
import tf.transformations as tx

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

    #plt.show()

def characterize_v2(xyz, rpy, err):
    xyz = np.float32(xyz)
    rpy = np.float32(rpy)
    fig = plt.figure()

    cm0 = matplotlib.cm.get_cmap('cool')
    cm = cm0(np.arange(cm0.N))
    cm[:,-1] = np.linspace(0.1, 1, cm0.N)
    cm = ListedColormap(cm)

    ax_p = fig.add_subplot(121, projection='3d')
    ax_p.set_xlabel('x')
    ax_p.set_ylabel('y')
    ax_p.set_zlabel('z')

    n = len(xyz)
    x,y,z = [xyz[:,i] for i in range(3)]
    R,P,Y = [rpy[:,i] for i in range(3)]
    s = ax_p.scatter(x,y,z, c=err, cmap=cm)
    fig.colorbar(s, ax=ax_p)

    ax_q = fig.add_subplot(122, projection='3d')
    ax_q.set_xlabel('R')
    ax_q.set_ylabel('P')
    ax_q.set_zlabel('Y')

    s = ax_q.scatter(R,P,Y, c=err, cmap=cm)
    fig.colorbar(s, ax=ax_q)

def plot_error(ax):
    e = np.loadtxt('/tmp/err.csv')
    ax.plot(e)
    plt.title('DH Parameter Error Over Time')
    plt.grid()
    plt.xlabel('Step')
    plt.ylabel('DH Parameter MSE')
    #plt.show()

def asub(a, b):
    """ Angular Residual """
    dx = np.subtract(a,b)
    return np.arctan2(np.sin(dx), np.cos(dx))

def main():
    """
    Requires:
        - '/tmp/err.csv' : DH Error over time
        - '/tmp/dhn.csv' : Nominal DH Parameter
        - '/tmp/dhf.csv' : Final DH Parameter

    As produced by dh_calibrator_ros.py after optimization.
    """

    _dh0 = np.loadtxt('/tmp/dhn.csv')
    _dh1 = np.loadtxt('/tmp/dhf.csv')

    print ('Final Error : ', _dh0 - _dh1)
    #_dh1 = np.concatenate((_dh1, _dh0[:, -1, np.newaxis]), axis=-1)

    xyz = []
    rpy = []
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
        xyz.append(txn0)
        rpy.append(rxn0)

    d_xyz = np.float32(d_xyz)
    d_rpy = np.float32(d_rpy)
    err = np.linalg.norm(np.concatenate([d_xyz, d_rpy], axis=-1), axis=-1)

    d_rpy = np.rad2deg(np.float32(d_rpy))

    print('d_xyz', np.mean(np.abs(d_xyz), axis=0))
    print('d_rpy', np.mean(np.abs(d_rpy), axis=0))
    characterize(d_xyz, d_rpy)
    characterize_v2(xyz, rpy, err)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_error(ax)

    plt.show()

if __name__ == "__main__":
    main()
