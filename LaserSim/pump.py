from LaserSim.utilities import numres, c, h, create_save_path, plot_function, integ, set_plot_params, generate_pulse
import numpy as np

set_plot_params()

class Pump():
    def __init__(self, intensity=30, # [kW/cm²]
                       duration=2, # [ms] 
                       wavelength=940, # [nm] 
                       bandwidth=0, # [nm]
                       gauss_order=1, # order of the gaussian pulse
                       resolution=numres,
                       spectral_type="rect"
                ):
        self.intensity = intensity*1e7   # [W/m²]
        self.duration = duration*1e-3    # [s]
        self.wavelength = wavelength*1e-9 # [m]
        self.dt = self.duration / resolution
        self.t_axis = np.linspace(0, self.duration, resolution)
        self.fluence = self.intensity * self.duration

        # Generate the spectral pump pulse
        self.seed_type = spectral_type
        self.bandwidth = bandwidth*1e-9
        self.seedres = resolution

        if self.bandwidth > 0: self.generate_pump_pulse(gauss_order)

    def generate_pump_pulse(self, gauss_order):
        if self.seed_type == 'rect': self.signal_length = 1.5
        elif self.seed_type == 'gauss': self.signal_length = 12/(8-5/gauss_order)
        elif self.seed_type == 'lorentz': self.signal_length = 10

        self.lambdas, self.pulse, self.dlambda = generate_pulse(self, self.bandwidth, center=self.wavelength)
        self.pulse *= 1/integ(self.pulse, self.dlambda)[-1]
    
    def __repr__(self):
        return(
        f"Pump:\n"
        f"- intensity = {self.intensity*1e-7} kW/cm² \n"
        f"- duration = {self.duration*1e3} ms \n"
        f"- wavelength = {self.wavelength*1e9:.2f} nm\n"
        F"- fluence = {self.fluence*1e-4} J/cm²\n\n"
        )

def plot_pump_pulse(pump, save=False, save_path=None, xlim=(-np.inf,np.inf), ylim=(-np.inf,np.inf)):
    """Plot the saturation intensity of a crystal."""
    if pump.bandwidth == 0:
        print("No spectral pump pulse generated, please specify a bandwidth > 0")
        return

    x = pump.lambdas*1e9
    y = pump.pulse
    xlabel = "wavelength in nm"
    ylabel = "normalized amplitude"
    legend = f"F = {integ(pump.pulse, pump.dlambda)[-1]*pump.fluence*1e-4:.3g} J/cm²"
    title= "Spectral pump pulse"
    fname = f"Pump_Spectral_{pump.wavelength*1e9}nm_{pump.bandwidth*1e9:.1f}nm.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, None, save, path, xlim, ylim)

if __name__ == "__main__":
    pump = Pump(bandwidth=10)
    print(pump)

    plot_pump_pulse(pump)