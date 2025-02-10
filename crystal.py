from utilities import numres, h, c
import json
import numpy as np
import os
import matplotlib.pyplot as plt


Folder = os.path.dirname(os.path.abspath(__file__))


class Crystal():
    def __init__(self, file="YbCaF2/basedata.json"):
        self.length = 1e-2      # [m]
        self.temperature = 300  # [K]
        self.tau_f = 0.95e-3       # [s]
        self.material = "YbCaF2"
        self.doping_concentration = 6e26  # [m^-3]
        self.inversion = np.zeros(numres)
        self.table_sigma_a = np.loadtxt(os.path.join(Folder, "material_database", "YbCaF2", "CaF300Ka.txt"))
        self.table_sigma_e = np.loadtxt(os.path.join(Folder, "material_database", "YbCaF2", "CaF300Kf.txt"))
        self.z_axis = np.linspace(0, self.length, numres)
        self.dz = self.length / numres

    def load_txt(self, file="YbCaF2/basedata.json"):
        self.dict = json.loads(file)
    
    def sigma_a(self,lambd):
        return np.interp(lambd*1e9, self.table_sigma_a[:,0], self.table_sigma_a[:,1])*1e-4
    
    def sigma_a(self, lambd):
        if lambd < 1000e-9:
            return 0.8E-24
        else:
            return 1.1e-25
    
    def sigma_e(self,lambd):
        return np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])*1e-4
    
    def sigma_e(self, lambd):
        if lambd < 1000e-9:
            return 0.16E-24
        else:
            return 2.3E-24
        
    def beta_eq(self,lambd):
        return self.sigma_a(lambd)/(self.sigma_a(lambd)+self.sigma_e(lambd))

    def __repr__(self):
        txt = f"Crystal:\nmaterial= {self.material}\nlength = {self.length*1e3} mm \ntau_f = {self.tau_f*1e3} ms \ndoping_concentration = {self.doping_concentration*1e-6} cm^-3"
        return txt

if __name__ == "__main__":
    p1 = Crystal()
    print(p1)
    print(p1.sigma_e(1030e-9))

    plt.figure()
    plt.plot(p1.table_sigma_e[:,0],p1.table_sigma_e[:,1])
    plt.show()

