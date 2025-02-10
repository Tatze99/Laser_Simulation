from utilities import numres, h, c
import numpy as np
import matplotlib.pyplot as plt

class Seed():
    def __init__(self):
        self.duration = 5e-9    # [s]
        self.wavelength = 1030e-9   # [m]
        self.fluence = 100     # [J/m²]
        self.gauss_order = 2
        self.seed_type = "gauss"
        self.seedres = 200
        self.stepsize = 2*self.duration / self.seedres
        self.time, self.pulse = self.pulse_gen()

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)
        t = np.linspace(-(self.seedres)/2, (self.seedres)/2, self.seedres)*self.stepsize

        # if self.seedtype == 'rect' or n==0:
        #     k1 = int((self.seed_sample_length - self.tau_seed) / (2 * stepsize))
        #     k2 = int((self.seed_sample_length + self.tau_seed) / (2 * stepsize))
        #     pulse[k1:k2] = self.F_in / h / self.nu_l / c / self.tau_seed

        if self.seed_type == 'gauss':
            pulse = np.exp( -(t / self.duration * 2) ** (2*self.gauss_order))
            pulse = self.fluence / h / c * self.wavelength / c / np.sum(pulse) / self.stepsize * pulse

        return t, pulse
    
    def __repr__(self):
        txt = f"Seed pulse:\nduration= {self.duration*1e9} ns\nwavelength = {self.wavelength*1e9} nm \nfluence = {self.fluence*1e4} J/cm²"
        return txt

if __name__ == "__main__":
    p1 = Seed()
    print(p1)
    plt.figure()
    # plt.plot(p1.time, p1.pulse)
    plt.show()