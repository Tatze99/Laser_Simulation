import matplotlib
import numpy as np
import matplotlib.pyplot as plt 
import os
import time

# define speed of light and planck's constant
c = 3e8 # [m/s]
h = 6.626e-34 # [Js]

# numerical resolution
numres = 300
LaserSimFolder = os.path.dirname(os.path.abspath(__file__))
LaserSimFolder = os.path.abspath(os.path.join(LaserSimFolder, os.pardir))

PLOT_DEFAULTS = dict(
    save=False,
    save_path=None,
    save_data=False,
    show_title=True,
    axis=None,
)

UNIT_TABLE = {
    "m": {
        "scale_to_m": 1.0,
        "label": "m"
    },
    "nm": {
        "scale_to_m": 1e-9,
        "label": "nm"
    },
    "um": {      # safer than µ in JSON
        "scale_to_m": 1e-6,
        "label": "µm"
    }
}

def set_plot_params():
    plt.rcParams["figure.figsize"] = (8,4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams['grid.linewidth'] = 0.5  # Adjust the value to make it thinner
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    plt.rc('font', family='serif')
    plt.rc('font', serif='Times New Roman')
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# integration routines: integ(x) = integral(0, x, y dx')
# convention integ: 1D
# f(t, z) -> 2D Matrix N x M with N in time und M in space
def integ(y, dx):
    """
    integrates a 1D array along its axis using the trapezian rule
    """
    trapezoids = (y[1:] + y[:-1]) / 2
    integral = np.cumsum(trapezoids) * dx
    return np.insert(integral, 0, 0)
    
def t_integ(mat, dt):
    """
    integrates the 2D matrix along the t-dimension, along axis 0
    """
    # average of adjacent rows
    trapezoids = (mat[1:,:] + mat[:-1,:]) / 2  
    
    # cumulative sum of trapezoid areas
    integral = np.cumsum(trapezoids, axis=0) * dt  
    
    # insert initial zero row
    return np.vstack([np.zeros((1, mat.shape[1])), integral])
    
def z_integ(mat, dz):
    """
    integrates the 2D matrix along the z-dimension, along axis 1
    """
    # average of adjacent columns
    trapezoids = (mat[:,1:] + mat[:,:-1]) / 2

    # cumulative sum of trapezoid areas
    integral = np.cumsum(trapezoids, axis=1) * dz

    # insert initial zero column
    return np.hstack([np.zeros((mat.shape[0], 1)), integral])

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

def plot_function(x, y, xlabel, ylabel, title=None, legends=None, axis=None, save=False, save_path=None, save_data=False, xlim = (-np.inf, np.inf), ylim = (-np.inf, np.inf), outer_legend = False, normalize=False, kwargs=None):
    """
    General function for plotting and saving figures.
    
    :param x: X-axis values
    :param y_list: List of Y-axis arrays (for multiple plots)
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param title: Plot title
    :param legends: List of legend labels (one per curve)
    :param save: Whether to save the plot
    :param save_path: Path to save the plot (if save=True)
    :param kwargs: Additional keyword arguments for the plot, can either be a dictionary or a list of dictionaries
    """
    if axis is None:
        fig, ax = plt.subplots()
    elif isinstance(axis, list):
        ax = axis[0]
        fig = axis[1]
    else:
        ax = axis

    # Plot multiple curves if needed
    if kwargs is None:
        kwargs = [dict(linestyle = "-", marker = None)] * len(y if isinstance(y, list) else [y])
    elif not isinstance(kwargs, list):
        kwargs = [kwargs] * len(y if isinstance(y, list) else [y])

    # Normalize x and y to lists
    if not isinstance(y, list):
        y = [y]
    if not isinstance(x, list):
        x = [x] * len(y)
    
    if normalize:
        y = [yi / np.max(yi) if np.max(yi) != 0 else yi for yi in y]

    for i, (x_elem, y_elem) in enumerate(zip(x, y)):
        kw = kwargs[i]
        label = legends[i] if legends and isinstance(legends, list) else (legends if i == 0 else None)
        ax.plot(x_elem, y_elem, label=label, **kw)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    bottom, top = ax.get_ylim()
    ax.set_ylim(max(bottom, ylim[0]), min(top, ylim[1]))
    left, right = ax.get_xlim()
    ax.set_xlim(max(left, xlim[0]), min(right, xlim[1]))

    if title:
        ax.set_title(title)

    if legends:
        if outer_legend:
            ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        else:
            ax.legend()

    if save and save_path:
        print(save_path)
        # plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
    
    if save_data and save_path:
        np.savetxt(os.path.splitext(save_path)[0]+".txt", np.vstack([x, y]).T, delimiter="\t", fmt="%.5e")
    
def create_save_path(save_path, fname):
    try: 
        path = os.path.join(save_path, fname)
    except:
        path = os.path.join(LaserSimFolder, "material_database", "plots", fname)

    return path

def generate_pulse(pulse, width, center=0.0, chirp_factor=1, x_min=None, x_max=None):
    """
    Generates a pulse with a given shape
    """
    x_min = center-pulse.signal_length*width*chirp_factor/2 if not x_min else x_min*1e-9
    x_max = center+pulse.signal_length*width*chirp_factor/2 if not x_max else x_max*1e-9

    dx = abs(x_max-x_min) / (pulse.seedres-1)
    y = np.zeros(pulse.seedres)
    x = np.linspace(x_min, x_max, pulse.seedres)

    if pulse.seed_type == 'gauss':
        y = np.exp( -np.log(2)*((x-center) / width * 2) ** (2*pulse.gauss_order))
        y *= 1/integ(y, dx)[-1]

    elif pulse.seed_type == 'lorentz':
        y = 1 / (1 + ((x-center) / width * 2)**2)
        y *= 1/integ(y, dx)[-1]

    else: # pulse.seed_type == 'rect' 
        y = np.ones(pulse.seedres) / width
        y = np.where((x-center) < -0.5*width, 0, y)
        y = np.where((x-center) >  0.5*width, 0, y)

    return x, y * pulse.fluence, dx

def generate_pulse_from_file(pulse, file_path, x_unit=1e0, delimiter="\t", x_min=None, x_max=None):
    """
    Generates a pulse from a file.
    """
    data = np.loadtxt(file_path, delimiter=delimiter)
    x_min = x_min if x_min is not None else min(data[:, 0])
    x_max = x_max if x_max is not None else max(data[:, 0])

    x = np.linspace(x_min, x_max, pulse.seedres)/ x_unit
    dx = x[1] - x[0]
    y = np.interp(x, data[:, 0]/ x_unit, data[:, 1])
    y *= 1/integ(y, dx)[-1]

    return x, y * pulse.fluence, dx

def integration_test(t_dim = numres, z_dim = numres, x_start=1, x_end=8, function=np.cos, analytical_integral=np.sin):
    """
    Test the integration routines
    """

    analytical_integral = analytical_integral(x_end) - analytical_integral(x_start)

    t = np.linspace(x_start, x_end, t_dim)
    z = np.linspace(x_start, x_end, z_dim)

    dt = t[1] - t[0]
    dz = z[1] - z[0]

    # create 2D sine matrix (N x M)
    matrix = np.zeros((t_dim, z_dim))
    matrix[:,0] = function(t)
    matrix[0,:] = function(z)

    time_integral = t_integ(matrix, dt)[-1,0]
    space_integral = z_integ(matrix, dz)[0,-1]

    time_1D_integral = integ(function(t), dt)[-1]
    space_1D_integral = integ(function(z), dz)[-1]

    print(f"Integration test for {function.__name__}(x) from {x_start} to {x_end} with the trapezoid rule:")
    print(f"{time_integral} : time integration")
    print(f"{time_1D_integral} : time 1D integration")
    print(f"{space_integral} : space integration")
    print(f"{space_1D_integral} : space 1D integration")
    print(f"{analytical_integral} : analytical integration\n")
    print(f"{np.sum(function(t)[:-1])*dt} : left Riemann sum")
    print(f"{np.sum(function(t)[1:])*dt} : right Riemann sum\n\n")

if __name__ == "__main__":
    integration_test()
    integration_test(function=np.exp, analytical_integral=np.exp)