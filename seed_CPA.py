from utilities import numres, h, c, integ
import numpy as np
import matplotlib.pyplot as plt

class Seed_CPA():
    def __init__(self, bandwidth = 30, wavelength = 1030, fluence = 1e-6, seed_type = "rect"):
        self.bandwidth = bandwidth*1e-9     # [m]
        self.wavelength = wavelength*1e-9   # [m]
        self.fluence = fluence*1e4          # [J/m²]
        self.gauss_order = 2
        self.seed_type = seed_type
        self.seedres = 250
        
        self.spectral_fluence = self.pulse_gen()

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)

        if self.seed_type == 'gauss':
            self.dlambda = 2*self.bandwidth / (self.seedres-1)
            self.lambdas = np.linspace(self.wavelength-self.bandwidth,self.wavelength+self.bandwidth, self.seedres)
            pulse = np.exp( -((self.lambdas-self.wavelength) / self.bandwidth * 2) ** (2*self.gauss_order))
            pulse *= 1/integ(pulse, self.dlambda)[-1]
        
        elif self.seed_type == 'rect':
            self.dlambda = self.bandwidth / (self.seedres-1)
            self.lambdas = np.linspace(self.wavelength-self.bandwidth/2,self.wavelength+self.bandwidth/2, self.seedres)
            pulse = np.ones(self.seedres) / self.bandwidth

        pulse *= self.fluence
        return pulse
    
    
    def __repr__(self):
        txt = f"Seed CPA pulse:\nbandwidth = {self.bandwidth*1e9:.2f} nm (FWHM)\nwavelength = {self.wavelength*1e9:.2f} nm \nfluence = {self.fluence*1e-4} J/cm²\npulse type = '{self.seed_type}'\n\n"
        return txt

if __name__ == "__main__":
    p1 = Seed_CPA(seed_type = "rect")

    print(p1)
    plt.figure()
    plt.plot(p1.lambdas*1e9, p1.spectral_fluence*1e-4*1e-9)
    plt.xlabel("wavlength in nm")
    plt.ylabel("spectral fluence in J/cm²/nm")