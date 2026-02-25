from LaserSim.utilities import h, c, integ, set_plot_params, plot_function, create_save_path, generate_pulse, generate_pulse_from_file
import numpy as np
import matplotlib.pyplot as plt
import os
set_plot_params()

class Seed_CPA():
    def __init__(self, wavelength = 1030, bandwidth = 30, fluence = 1e-4, seed_type = "gauss", gauss_order = 1, custom_file = None, custom_file_delimiter="\t", custom_file_xunit=1e0, resolution = 250, lambda_min=None, lambda_max=None, chirp="positive"):
        """
        Docstring for __init__
        
        :param wavelength: Central wavelength in nm
        :param bandwidth: Bandwidth (FWHM) in nm
        :param fluence: Input fluence in J/cm²
        :param gauss_order: Order of the Gaussian pulse (only for seed_type = "gauss")
        :param seed_type: Shape of the temporal seed pulse, can be "gauss", "rect" or "lorentz"
        :param resolution: Number of points for the spectral seed pulse
        :param lambda_min: lambda_min in nm
        :param lambda_max: lambda_max in nm
        :param custom_file: custom pulse file path
        :param custom_file_delimiter: delimiter of the custom pulse file 
        :param custom_file_xunit: unit of the x-axis in the custom pulse file (default is 1e0 = 1 nm)
        """
        self.bandwidth = bandwidth*1e-9     # [m]
        self.wavelength = wavelength*1e-9   # [m]
        self.fluence = fluence*1e4          # [J/m²]
        self.gauss_order = gauss_order
        self.seed_type = seed_type
        self.seedres = resolution
        self.chirp = chirp
        self.CPA = True # boolean to indicate that this is a CPA seed pulse
        
        if seed_type == 'rect': self.signal_length = 1
        elif seed_type == 'gauss': self.signal_length = 2
        elif seed_type == 'lorentz': self.signal_length = 4
        else:
            self.signal_length = 1
            print(f"Warning: seed_type '{seed_type}' not recognized, using default 'rect'")

        if self.chirp == "positive": chirp_factor = 1
        elif self.chirp == "negative": chirp_factor = -1
        else: chirp_factor = 1

        if custom_file:
            self.lambdas, self.spectral_fluence, self.dlambda = generate_pulse_from_file(self, custom_file, delimiter=custom_file_delimiter, x_unit=custom_file_xunit, x_min=lambda_min, x_max=lambda_max)
        else:
            self.lambdas, self.spectral_fluence, self.dlambda = generate_pulse(self, self.bandwidth, center=self.wavelength, chirp_factor=chirp_factor, x_min=lambda_min, x_max=lambda_max)


    def __repr__(self):
        return(
        f"Seed CPA pulse:\n"
        f"- bandwidth = {self.bandwidth*1e9:.2f} nm (FWHM)\n"
        f"- wavelength = {self.wavelength*1e9:.2f} nm \n"
        f"- fluence = {self.fluence*1e-4} J/cm²\n"
        f"- pulse type = '{self.seed_type}'\n\n"
        )

def plot_seed_pulse(seed, save=False, save_path=None, save_data=False, xlim=(-np.inf, np.inf), ylim=(0,np.inf), show_title=True, axis=None):
    """Plot the seed CPA pulse."""
    x = seed.lambdas*1e9
    y = seed.spectral_fluence*1e-4*1e-9
    xlabel = "wavelength in nm"
    ylabel = "spectral fluence in J/cm²/nm"
    legend= f"F = {integ(seed.spectral_fluence, seed.dlambda)[-1]/1e4:.3g} J/cm²"
    title = "Spectral seed pulse" if show_title else None
    fname = f"Seed_Spectrum_{seed.wavelength*1e9}nm_{seed.bandwidth*1e9:.1f}nm.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, xlim, ylim)

if __name__ == "__main__":
    seed = Seed_CPA()

    print(seed)
    plot_seed_pulse(seed, save=False)