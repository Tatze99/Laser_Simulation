from utilities import numres, h, c, integ, set_plot_params
import numpy as np
import matplotlib.pyplot as plt
import os
set_plot_params()
Folder = os.path.dirname(os.path.abspath(__file__))

class Seed_CPA():
    def __init__(self, bandwidth = 30, wavelength = 1030, fluence = 1e-6, seed_type = "gauss", gauss_order = 1):
        self.bandwidth = bandwidth*1e-9     # [m]
        self.wavelength = wavelength*1e-9   # [m]
        self.fluence = fluence*1e4          # [J/m²]
        self.gauss_order = gauss_order
        self.seed_type = seed_type
        self.seedres = 250
        
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

        pulse *= self.fluence
        return pulse
    
    
    def __repr__(self):
        return(
        f"Seed CPA pulse:\n"
        f"- bandwidth = {self.bandwidth*1e9:.2f}nm (FWHM)\n"
        f"- wavelength = {self.wavelength*1e9:.2f}nm \n"
        f"- fluence = {self.fluence*1e-4}J/cm²\n"
        f"- pulse type = '{self.seed_type}'\n\n"
        )

def plot_seed_pulse(seed, save=False):
    """
    Plot the seed pulse spectrum.
    """
    plt.figure()
    plt.plot(seed.lambdas*1e9, seed.spectral_fluence*1e-4*1e-9, label=f"F = {integ(seed.spectral_fluence, seed.dlambda)[-1]/1e4:.1e} J/cm²")
    plt.legend()
    plt.xlabel("wavlength in nm")
    plt.ylabel("spectral fluence in J/cm²/nm")
    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"Seed_Spectrum_{seed.wavelength*1e9}nm_{seed.bandwidth*1e9:.1f}nm.pdf"))

if __name__ == "__main__":
    seed = Seed_CPA(seed_type = "gauss", gauss_order = 1)

    print(seed)
    plot_seed_pulse(seed, save=True)