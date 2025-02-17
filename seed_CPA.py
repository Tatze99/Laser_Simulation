from utilities import numres, h, c
import numpy as np
import matplotlib.pyplot as plt

class Seed_CPA():
    def __init__(self, bandwidth = 10e-9, wavelength = 1030e-9, fluence = 100, seed_type = "gauss"):
        self.bandwidth = bandwidth    # [m]
        self.wavelength = wavelength   # [m]
        self.fluence = fluence     # [J/m²]
        self.GDD = 1.3e-24        # [s²]
        self.gauss_order = 2
        self.seed_type = seed_type
        self.seedres = 500
        self.dlambda = 2*self.bandwidth / self.seedres
        self.dt = self.GDD*2*np.pi*c/self.wavelength**2
        self.time, self.lambdas, self.pulse = self.pulse_gen()
        self.spectral_fluence = self.fluence*np.ones(self.seedres)*1/(self.seedres-1)
        self.lambdas = np.linspace(1000e-9,1060e-9, self.seedres)

    def temporal_delay(self, lamb):
        angular_frequency = 2*np.pi*c/(self.wavelength)
        return self.GDD*angular_frequency*(1-lamb/self.wavelength)

    def spectral_delay(self, tau):
        angular_frequency = 2*np.pi*c/(self.wavelength)
        return self.wavelength*(1-tau/(self.GDD*angular_frequency))

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)
        time = np.linspace(-(self.seedres)/2, (self.seedres)/2, self.seedres)*self.dt
        lambdas = self.spectral_delay(time)+self.wavelength

        if self.seed_type == 'gauss':
            pulse = np.exp( -((lambdas-self.wavelength) / self.bandwidth * 2) ** (2*self.gauss_order))
            # pulse = self.fluence / h / c * self.wavelength / c / np.sum(pulse) / self.stepsize * pulse

        return time, lambdas, pulse
    
    
    def __repr__(self):
        txt = f"Seed pulse:\nduration= {self.duration*1e9} ns\nwavelength = {self.wavelength*1e9} nm \nfluence = {self.fluence*1e4} J/cm²"
        return txt

if __name__ == "__main__":
    p1 = Seed_CPA()
    # print(p1)
    plt.figure()
    print(p1.pulse)
    plt.plot(p1.lambdas, p1.pulse)
    plt.show()