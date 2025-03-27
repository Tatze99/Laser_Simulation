from LaserSim.utilities import numres
import numpy as np

class Pump():
    def __init__(self, intensity=30, # [kW/cm²]
                       duration=2, # [ms] 
                       wavelength=940, # [nm] 
                       resolution=numres
                ):
        self.intensity = intensity*1e7   # [W/m²]
        self.duration = duration*1e-3    # [s]
        self.wavelength = wavelength*1e-9 # [m]
        self.dt = self.duration / resolution
        self.t_axis = np.linspace(0, self.duration, resolution)
        self.fluence = self.intensity * self.duration

    def __repr__(self):
        return(
        f"Pump:\n"
        f"- intensity = {self.intensity*1e-7} kW/cm² \n"
        f"- duration = {self.duration*1e3} ms \n"
        f"- wavelength = {self.wavelength*1e9:.2f} nm\n"
        F"- fluence = {self.fluence*1e-4} J/cm²\n\n"
        )

if __name__ == "__main__":
    pump = Pump()
    print(pump)