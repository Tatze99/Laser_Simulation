import numpy as np

# define speed of light and planck's constant
c = 3e8 # [m/s]
h = 6.626e-34 # [Js]

# numerical resolution
numres = 200

# integration routines: integ(x) = integral(0, x, y dx')
# convention integ: 1D
# f(t, z) -> 2D Matrix N x M with N in time und M in space

def integ(y, dx):
    os = np.cumsum(y)
    us = np.array(os)
    us[1::] = os[0:-1]
    us[0] = 0
    return (os + us) / 2 * dx
    

def t_integ(mat, dt):
    os = np.cumsum(mat, axis=0)
    us = np.array(os)
    us[1::, :] = os[0:-1,:]
    us[0,:] = 0
    return (os + us) / 2 * dt

def z_integ(mat, dz):
    """
    integrates the matrix along the z-dimension, along axis 1
    """
    os = np.cumsum(mat, axis=1)
    us = np.array(os)
    us[:, 1::] = os[:,0:-1]
    us[:,0] = 0
    return (os + us) / 2 * dz