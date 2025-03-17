import numpy as np
import matplotlib.pyplot as plt 


# define speed of light and planck's constant
c = 3e8 # [m/s]
h = 6.626e-34 # [Js]

# numerical resolution
numres = 300

def set_plot_params():
    plt.rcParams["figure.figsize"] = (8,4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust the value to make it thinner
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

# integration routines: integ(x) = integral(0, x, y dx')
# convention integ: 1D
# f(t, z) -> 2D Matrix N x M with N in time und M in space
def integ(y, dx):
    """
    integrates a 1D array y along the its axis using the trapezian rule"""
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

def fourier_filter(sigma, filter_width):
    """
    fourier filter for the given sigma array
    fourier filter cutoff: 0 ... 1 (0: no filter, 1: all frequencies are filtered)
    """
    if filter_width == 0:
        return sigma[:,1]
    
    fft = np.fft.fft(sigma[:,1])
    fft_filter = np.ones_like(sigma[:,1])
    mid_index = int(len(fft_filter)/2)
    fft_filter[mid_index-int(filter_width*mid_index):mid_index+int(filter_width*mid_index)] = 0
    
    fft_neu = fft*fft_filter
    filtered_sigma = np.fft.ifft(fft_neu).real
    
    return filtered_sigma

def moving_average(x, window_size):
    """
    Smooths the given array x using a centered moving average.

    Parameters:
        x (list or numpy array): The input array to be smoothed.
        window_size (int): The size of the centered moving average window.

    Returns:
        smoothed_array (numpy array): The smoothed array.
    """
    # Ensure the window_size is even
    if window_size % 2 == 0:
        half_window = window_size // 2
    else:
        half_window = (window_size - 1) // 2

    if window_size <= 1:
        return x

    half_window = window_size // 2
    cumsum = np.cumsum(x)

    # Calculate the sum of elements for each centered window
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    centered_sums = cumsum[window_size - 1:-1]

    # Divide each sum by the window size to get the centered moving average
    smoothed_array = centered_sums / window_size

    # Pad the beginning and end of the smoothed array with the first and last values of x
    first_value = np.repeat(x[0], half_window)
    last_value = np.repeat(x[-1], half_window)
    smoothed_array = np.concatenate((first_value, smoothed_array, last_value))

    return smoothed_array