import numpy as np

# define speed of light and planck's constant
c = 3e8 # [m/s]
h = 6.626e-34 # [Js]

# numerical resolution
numres = 300

# integration routines: integ(x) = integral(0, x, y dx')
# convention integ: 1D
# f(t, z) -> 2D Matrix N x M with N in time und M in space

def integ(y, dx):
    upper_sum = np.cumsum(y[1:])
    lower_sum = np.cumsum(y[:-1])
	
    integral = (upper_sum + lower_sum) / 2 * dx
    return np.insert(integral, 0, 0)
    

def t_integ(mat, dt):
    """
    integrates the matrix along the t-dimension, along axis 0
    """
    upper_sum = np.cumsum(mat[1:,:], axis=0)
    lower_sum = np.cumsum(mat[:-1,:], axis=0)
    lower_sum = np.array(upper_sum)

    integral = (upper_sum + lower_sum) / 2 * dt
    return np.insert(integral, 0, 0, axis=0)

def z_integ(mat, dz):
    """
    integrates the matrix along the z-dimension, along axis 1
    """
    upper_sum = np.cumsum(mat[:,1:], axis=1)
    lower_sum = np.cumsum(mat[:,:-1], axis=1)
    lower_sum = np.array(upper_sum)

    integral = (upper_sum + lower_sum) / 2 * dz
    return np.insert(integral, 0, 0, axis=1)