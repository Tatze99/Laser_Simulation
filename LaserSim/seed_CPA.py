from LaserSim.utilities import h, c, integ, set_plot_params, plot_function
import numpy as np
import matplotlib.pyplot as plt
import os
set_plot_params()
Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

class Seed_CPA():
    def __init__(self, wavelength = 1030, bandwidth = 30, fluence = 1e-4, seed_type = "gauss", gauss_order = 1, custom_file = None, resolution = 250):
        self.bandwidth = bandwidth*1e-9     # [m]
        self.wavelength = wavelength*1e-9   # [m]
        self.fluence = fluence*1e4          # [J/m²]
        self.gauss_order = gauss_order
        self.seed_type = seed_type
        self.seedres = resolution
        self.custom_file = custom_file
        
        self.spectral_fluence = self.pulse_gen()

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)

        if self.seed_type == 'gauss':
            self.dlambda = 2*self.bandwidth / (self.seedres-1)
            self.lambdas = np.linspace(self.wavelength-self.bandwidth,self.wavelength+self.bandwidth, self.seedres)
            pulse = np.exp( -np.log(2)*((self.lambdas-self.wavelength) / self.bandwidth * 2) ** (2*self.gauss_order))
            pulse *= 1/integ(pulse, self.dlambda)[-1]
        
        elif self.seed_type == 'rect':
            self.dlambda = self.bandwidth / (self.seedres-1)
            self.lambdas = np.linspace(self.wavelength-self.bandwidth/2,self.wavelength+self.bandwidth/2, self.seedres)
            pulse = np.ones(self.seedres) / self.bandwidth
            pulse = np.where(self.lambdas < self.wavelength-0.5*self.bandwidth, 0, pulse)
            pulse = np.where(self.lambdas > self.wavelength+0.5*self.bandwidth, 0, pulse)
        
        elif self.seed_type == 'custom':
            self.dlambda = 2*self.bandwidth / (self.seedres-1)
            self.lambdas = np.linspace(self.wavelength-self.bandwidth,self.wavelength+self.bandwidth, self.seedres)
            try:
                pulse = np.loadtxt(self.custom_file)
                factor = 1e-9 if np.max(pulse[:,0]) > 1 else 1
                pulse = np.interp(self.lambdas, pulse[:,0]*factor, pulse[:,1])
                pulse -= np.min(abs(pulse))
                pulse *= 1/integ(pulse, self.dlambda)[-1]
            except:
                print("Custom file not found")

        pulse *= self.fluence
        return pulse
    
    
    def __repr__(self):
        return(
        f"Seed CPA pulse:\n"
        f"- bandwidth = {self.bandwidth*1e9:.2f} nm (FWHM)\n"
        f"- wavelength = {self.wavelength*1e9:.2f} nm \n"
        f"- fluence = {self.fluence*1e-4} J/cm²\n"
        f"- pulse type = '{self.seed_type}'\n\n"
        )

def plot_seed_pulse(seed, save=False, save_path=None, xlim=(1000,1060), ylim=(0,np.inf)):
    """Plot the saturation intensity of a crystal."""
    x = seed.lambdas*1e9
    y = seed.spectral_fluence*1e-4*1e-9
    xlabel = "wavelength in nm"
    ylabel = "spectral fluence in J/cm²/nm"
    legend= f"F = {integ(seed.spectral_fluence, seed.dlambda)[-1]/1e4:.3g} J/cm²"
    title = "Spectral seed pulse"
    path = save_path or os.path.join(Folder, "material_database","plots", f"Seed_Spectrum_{seed.wavelength*1e9}nm_{seed.bandwidth*1e9:.1f}nm.pdf")

    plot_function(x, y, xlabel, ylabel, title, legend, save, path, xlim, ylim)

if __name__ == "__main__":
    seed = Seed_CPA()
 
    print(seed)
    plot_seed_pulse(seed, save=False)