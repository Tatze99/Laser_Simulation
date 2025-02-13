from utilities import numres
import numpy as np

class Pump():
    def __init__(self, intensity=30, # [kW/cm²]
                       duration=2, # [ms] 
                       wavelength=940 # [nm] 
                ):
        self.intensity = intensity*1e7   # [W/m²]
        self.duration = duration*1e-3    # [s]
        self.wavelength = wavelength*1e-9 # [m]
        self.dt = self.duration / numres
        self.t_axis = np.linspace(0, self.duration, numres)

    def __repr__(self):
        txt = f"Pump:\nintensity = {self.intensity*1e-7} kW/cm² \nduration = {self.duration*1e3} ms \nwavelength = {self.wavelength*1e9} nm"
        return txt

if __name__ == "__main__":
    p1 = Pump()
    print(p1)