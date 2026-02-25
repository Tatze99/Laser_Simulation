from LaserSim.utilities import h, c, integ, set_plot_params, plot_function, create_save_path, generate_pulse, generate_pulse_from_file
import numpy as np
import matplotlib.pyplot as plt
import os
set_plot_params()

class Seed():
    def __init__(self, fluence = 1e-4, duration = 5, wavelength = 1030, gauss_order = 1, seed_type = "gauss", resolution = 200, t_min=None, t_max=None, custom_file = None, custom_file_delimiter="\t", custom_file_xunit=1e0, ):
        """
        Docstring for __init__
        
        :param fluence: Input fluence in J/cm²
        :param duration: Pulse duration in ns
        :param wavelength: Central wavelength in nm
        :param gauss_order: Order of the Gaussian pulse (only for seed_type = "gauss")
        :param seed_type: Shape of the temporal seed pulse, can be "gauss", "rect" or "lorentz"
        :param resolution: Number of points for the temporal seed pulse
        :param t_min: start time in ns
        :param t_max: end time in ns
        :param custom_file: custom pulse file path
        :param custom_file_delimiter: delimiter of the custom pulse file 
        :param custom_file_xunit: unit of the x-axis in the custom pulse file (default is 1e0 = 1 ns)
        """
        self.duration = duration*1e-9     # [s]
        self.wavelength = wavelength*1e-9 # [m]
        self.fluence = fluence*1e4        # [J/m²]
        self.gauss_order = gauss_order
        self.seed_type = seed_type
        self.seedres = resolution
        self.CPA = False # boolean to indicate that this is not a CPA seed pulse

        if seed_type == 'rect': self.signal_length = 1.5
        elif seed_type == 'gauss': self.signal_length = 12/(8-5/gauss_order)
        elif seed_type == 'lorentz': self.signal_length = 10
        else:
            self.signal_length = 10
            print(f"Warning: seed_type '{seed_type}' not recognized, using default 'rect'")

        if custom_file:
            self.time, self.pulse, self.dt = generate_pulse_from_file(self, custom_file, delimiter=custom_file_delimiter, x_unit=custom_file_xunit, x_min=t_min, x_max=t_max)
        else:
            self.time, self.pulse, self.dt = generate_pulse(self, self.duration, x_min=t_min, x_max=t_max)

        self.pulse *= 1/ h / c * self.wavelength / c

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)
        t = np.linspace(-(self.seedres)/2, (self.seedres)/2, self.seedres)*self.dt

        if self.seed_type == 'gauss':
            pulse = np.exp( -np.log(2)*(t / self.duration * 2) ** (2*self.gauss_order))
            pulse *= 1/integ(pulse, self.dt)[-1]

        elif self.seed_type == 'lorentz':
            pulse = 1 / (1 + (t / self.duration * 2)**2)
            pulse *= 1/integ(pulse, self.dt)[-1]
        
        elif self.seed_type == 'rect':
            pulse = np.ones(self.seedres) / self.duration
            pulse = np.where(t < -0.5*self.duration, 0, pulse)
            pulse = np.where(t >  0.5*self.duration, 0, pulse)

        pulse *= self.fluence / h / c * self.wavelength / c

        return t, pulse
    
    def __repr__(self):
        return(
        f"Seed CW pulse:\n"
        f"- duration = {self.duration*1e9} ns\n"
        f"- wavelength = {self.wavelength*1e9} nm \n"
        f"- fluence = {self.fluence*1e-4} J/cm²\n"
        f"- pulse type = '{self.seed_type}'\n\n"
        )

def plot_seed_pulse(seed, save=False, save_path=None, save_data=False, xlim=(-np.inf,np.inf), ylim=(-np.inf,np.inf), show_title=True, axis=None):
    """Plot the saturation intensity of a crystal."""
    x = seed.time*1e9
    y = seed.pulse
    xlabel = "time in ns"
    ylabel = "photon density $\\Phi$ in 1/m³"
    legend = f"F = {integ(seed.pulse, seed.dt)[-1]* c * h *c / seed.wavelength*1e-4:.3g} J/cm²"
    title= "Temporal seed pulse" if show_title else None
    fname = f"Seed_Temporal_{seed.wavelength*1e9}nm_{seed.duration*1e9:.1f}ns.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, xlim, ylim)

if __name__ == "__main__":
    seed = Seed()
    
    print(seed)
    plot_seed_pulse(seed)