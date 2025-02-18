from utilities import numres, h, c, integ
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams["axes.grid"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

class Seed():
    def __init__(self, fluence = 100, duration = 5e-9, wavelength = 1030e-9, gauss_order = 2, seed_type = "gauss"):
        self.duration = duration        # [s]
        self.wavelength = wavelength    # [m]
        self.fluence = fluence          # [J/m²]
        self.gauss_order = gauss_order
        self.seed_type = seed_type
        self.seedres = 200
        self.dt = 2*self.duration / (self.seedres-1)
        self.time, self.pulse = self.pulse_gen()

    def pulse_gen(self):
        pulse = np.zeros(self.seedres)
        t = np.linspace(-(self.seedres)/2, (self.seedres)/2, self.seedres)*self.dt

        if self.seed_type == 'gauss':
            pulse = np.exp( -(t / self.duration * 2) ** (2*self.gauss_order))
            pulse *= 1/integ(pulse, self.dt)[-1]
        
        elif self.seed_type == 'rect':
            pulse = np.ones(self.seedres) / self.duration
            pulse = np.where(t < -0.5*self.duration, 0, pulse)
            pulse = np.where(t >  0.5*self.duration, 0, pulse)

        pulse *= self.fluence / h / c * self.wavelength / c
        # if self.seed_type == 'gauss':
        #     pulse = np.exp( -(t / self.duration * 2) ** (2*self.gauss_order))
        #     pulse = self.fluence / h / c * self.wavelength / c / np.sum(pulse) / self.dt * pulse

        return t, pulse
    
    def __repr__(self):
        txt = f"Seed CW pulse:\nduration= {self.duration*1e9} ns\nwavelength = {self.wavelength*1e9} nm \nfluence = {self.fluence*1e-4} J/cm²\npulse type = '{self.seed_type}'\n\n"
        return txt

if __name__ == "__main__":
    seed = Seed()
    print(seed)
    plt.figure()
    plt.plot(seed.time*1e9, seed.pulse, label=f"F = {integ(seed.pulse, seed.dt)[-1]* c * h *c / seed.wavelength:.2f} J/m²")
    plt.legend()