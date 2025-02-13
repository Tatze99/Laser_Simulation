from utilities import numres, h, c
import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


Folder = os.path.dirname(os.path.abspath(__file__))


class Crystal():
    def __init__(self, material="YbCaF2", temperature="300"):
        # self.length = 1e-2      # [m]
        # self.tau_f = 0.95e-3       # [s]
        # self.doping_concentration = 6e26  # [m^-3]
        self.inversion = np.zeros(numres)
        self.temperature = temperature

        basedata_path = os.path.join(Folder,"material_database", material, "basedata.json")
        sigma_a_path = glob.glob(os.path.join(Folder, "material_database", material, f"*{temperature}Ka.txt"))
        sigma_e_path = glob.glob(os.path.join(Folder, "material_database", material, f"*{temperature}Kf.txt"))

        if sigma_a_path and sigma_e_path:
            self.table_sigma_a = np.loadtxt(sigma_a_path[0])
            self.table_sigma_e = np.loadtxt(sigma_e_path[0])
        else:
            raise FileNotFoundError(f"No file found for file pattern: {Folder} -> material_database -> {material} @ {temperature}K")

        self.load_basedata(basedata_path)

        self.z_axis = np.linspace(0, self.length, numres)
        self.dz = self.length / (numres-1)

    def load_basedata(self, filename):
        with open(filename, "r") as file:
            self.dict = json.load(file)
            self.length = self.dict["length"]
            self.tau_f = self.dict["tau_f"]
            self.doping_concentration = self.dict["N_dop"]
            self.name = self.dict["name"]

    
    # absorption cross section
    def sigma_a(self,lambd):
        return np.interp(lambd*1e9, self.table_sigma_a[:,0], self.table_sigma_a[:,1])*1e-4
        
    # emission cross section
    def sigma_e(self,lambd):
        return np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])*1e-4
    
    # equilibrium inversion
    def beta_eq(self,lambd):
        return self.sigma_a(lambd)/(self.sigma_a(lambd)+self.sigma_e(lambd))
    
    # Absorption coefficient
    def alpha(self,lambd):
        return self.doping_concentration * self.sigma_a(lambd)
    
    def small_signal_gain(self, lambd):
        Gain = np.exp(self.doping_concentration*(self.sigma_e(lambd)))

    def __repr__(self):
        txt = f"Crystal:\nmaterial = {self.name}\nlength = {self.length*1e3} mm \ntau_f = {self.tau_f*1e3} ms \nN_dop = {self.doping_concentration*1e-6} cm^-3"
        return txt


def plot_cross_sections(crystal):
    plt.figure(figsize=(8,4))
    plt.plot(crystal.table_sigma_a[:,0], crystal.table_sigma_a[:,1], label="absorption $\\sigma_a$")
    plt.plot(crystal.table_sigma_e[:,0], crystal.table_sigma_e[:,1], label="emission $\\sigma_e$")
    plt.xlabel("wavelength in nm")
    plt.ylabel("cross section in cm²")
    plt.title(f"{crystal.name} at {crystal.temperature}K")
    plt.legend()

if __name__ == "__main__":
    crystal = Crystal(material="YbYAG", temperature=100)
    crystal = Crystal(material="YbCaF2", temperature=300)
    print(crystal)

    plot_cross_sections(crystal)


