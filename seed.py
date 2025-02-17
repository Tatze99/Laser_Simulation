from utilities import numres, h, c, integ
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
        self.dt = 2*self.duration / (self.seedres-1)
        self.time, self.pulse = self.pulse_gen()

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)
        t = np.linspace(-(self.seedres)/2, (self.seedres)/2, self.seedres)*self.dt

        if self.seed_type == 'gauss':
            pulse = np.exp( -(t / self.duration * 2) ** (2*self.gauss_order))
            pulse *= 1/integ(pulse, self.dt)[-1]*self.fluence / h / c * self.wavelength / c
        
        elif self.seed_type == 'rect':
            pulse = np.ones(self.seedres) / self.duration
            pulse = np.where(self.time < -0.5*self.duration, 0, pulse)
            pulse = np.where(self.time >  0.5*self.duration, 0, pulse)
            pulse *= self.fluence / h / c * self.wavelength / c

        # if self.seed_type == 'gauss':
        #     pulse = np.exp( -(t / self.duration * 2) ** (2*self.gauss_order))
        #     pulse = self.fluence / h / c * self.wavelength / c / np.sum(pulse) / self.dt * pulse

        return t, pulse
    
    def __repr__(self):
        txt = f"Seed CW pulse:\nduration= {self.duration*1e9} ns\nwavelength = {self.wavelength*1e9} nm \nfluence = {self.fluence*1e-4} J/cm²\npulse type = '{self.seed_type}'\n\n"
        return txt

if __name__ == "__main__":
    p1 = Seed()
    print(p1)
    plt.figure()
    # plt.plot(p1.time, p1.pulse)
    plt.show()