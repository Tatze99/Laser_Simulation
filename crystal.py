from utilities import numres, h, c, moving_average, fourier_filter, set_plot_params
import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
set_plot_params()
# plt.rcParams["figure.figsize"] = (10,5)

Folder = os.path.dirname(os.path.abspath(__file__))

def logistic_function(x, a,b):
    return 1/(1+np.exp(-a*(x-b)))

class Crystal():
    def __init__(self, material="YbCaF2", temperature="300"):
        self.inversion = np.zeros(numres)
        self.temperature = temperature
        self.use_spline_interpolation = True
        self.material = material

        basedata_path = os.path.join(Folder,"material_database", material, "basedata.json")
        
        self.load_basedata(basedata_path)
        self.load_cross_sections(material)
        self.load_spline_interpolation()

        self.z_axis = np.linspace(0, self.length, numres)
        self.dz = self.length / (numres-1)
        self.inversion_end = None

    def load_basedata(self, filename):
        with open(filename, "r") as file:
            self.dict = json.load(file)
            self.length = self.dict["length"]
            self.tau_f = self.dict["tau_f"]
            self.doping_concentration = self.dict["N_dop"]
            self.name = self.dict["name"]

    def load_spline_interpolation(self):
        self.spline_sigma_a = CubicSpline(self.table_sigma_a[:,0], self.table_sigma_a[:,1], bc_type="clamped")
        self.spline_sigma_e = CubicSpline(self.table_sigma_e[:,0], self.table_sigma_e[:,1], bc_type="clamped")

    def load_cross_sections(self, material):
        sigma_a_path = glob.glob(os.path.join(Folder, "material_database", material, f"*{self.temperature}Ka.*"))
        sigma_e_path = glob.glob(os.path.join(Folder, "material_database", material, f"*{self.temperature}Kf.*"))

        if sigma_a_path and sigma_e_path:
            self.table_sigma_a = np.nan_to_num(np.loadtxt(sigma_a_path[0]))
            self.table_sigma_e = np.nan_to_num(np.loadtxt(sigma_e_path[0]))
            if self.table_sigma_a[1,0] < self.table_sigma_a[0,0]:
                self.table_sigma_a = np.flipud(self.table_sigma_a)
            if self.table_sigma_e[1,0] < self.table_sigma_e[0,0]: 
                self.table_sigma_e = np.flipud(self.table_sigma_e)
        else:
            raise FileNotFoundError(f"No file found for file pattern: {Folder} -> material_database -> {self.name} @ {self.temperature}K")
        
    def smooth_cross_sections(self, FF_filter, mov_average, lambda_min=980e-9):
        self.table_sigma_a[:,1] = fourier_filter(self.table_sigma_a, FF_filter)
        self.table_sigma_e[:,1] = fourier_filter(self.table_sigma_e, FF_filter)
        self.table_sigma_a[:,1] = moving_average(self.table_sigma_a[:,1], mov_average)
        self.table_sigma_e[:,1] = moving_average(self.table_sigma_e[:,1], mov_average)

        self.McCumber_absorption(lambda_min)
        self.load_spline_interpolation()
        
    # absorption cross section
    def sigma_a(self,lambd):
        if self.use_spline_interpolation:
            return self.spline_sigma_a(lambd*1e9)*1e-4
        else:
            return np.interp(lambd*1e9, self.table_sigma_a[:,0], self.table_sigma_a[:,1])*1e-4
        
    # emission cross section
    def sigma_e(self,lambd):
        if self.use_spline_interpolation:
            return self.spline_sigma_e(lambd*1e9)*1e-4
        else:
            return np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])*1e-4
    
    # equilibrium inversion
    def beta_eq(self,lambd):
        return self.sigma_a(lambd)/(self.sigma_a(lambd)+self.sigma_e(lambd))
    
    # Absorption coefficient
    def alpha(self,lambd):
        return self.doping_concentration * self.sigma_a(lambd)
    
    def small_signal_gain(self, lambd, beta):
        Gain = np.exp(self.doping_concentration*(self.sigma_e(lambd)*beta-self.sigma_a(lambd)*(1-beta))*self.length)
        return Gain
    
    def McCumber_absorption(self, lambda_min=980e-9):
        lambd = self.table_sigma_a[:,0]*1e-9
        min_index = np.argmin(np.abs(lambd-lambda_min))
        beta_eq = self.beta_eq(lambd)
        params, _ = curve_fit(logistic_function, lambd, beta_eq, p0=[1,980e-9])
        interpolated_sigma_e = np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])
        self.table_sigma_a[min_index:,1] = interpolated_sigma_e[min_index:] * logistic_function(lambd[min_index:], *params)/(1-logistic_function(lambd[min_index:], *params))
        self.load_spline_interpolation()



    def __repr__(self):
        txt = f"Crystal:\nmaterial = {self.name}\nlength = {self.length*1e3} mm \ntau_f = {self.tau_f*1e3} ms \nN_dop = {self.doping_concentration*1e-6} cm^-3\nsigma_a(940nm) = {self.sigma_a(940e-9)*1e4:.3e}cm²\nsigma_e(940nm) = {self.sigma_e(940e-9)*1e4:.3e}cm²\n\n"
        return txt

# =============================================================================
# Display of results
# =============================================================================

def plot_cross_sections(crystal, save=False):
    plt.figure()
    plt.plot(crystal.table_sigma_a[:,0], crystal.table_sigma_a[:,1], label="absorption $\\sigma_a$")
    plt.plot(crystal.table_sigma_e[:,0], crystal.table_sigma_e[:,1], label="emission $\\sigma_e$")
    plt.xlabel("wavelength in nm")
    plt.ylabel("cross section in cm²")
    plt.title(f"{crystal.name} at {crystal.temperature}K")
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_cross_sections.pdf"))

def plot_small_signal_gain(crystal, beta, lam_min = 1000, lam_max = 1060, save=False):
    plt.figure()
    lambd = np.linspace(lam_min*1e-9, lam_max*1e-9,100)

    # make multiple plots if beta is an array!
    if isinstance(beta, (list, tuple, np.ndarray)): 
        for b in beta:
            Gain = crystal.small_signal_gain(lambd, b)**2*(1-0.078)
            plt.plot(lambd*1e9, Gain, label=f"$\\beta$ = {b:.2f}")
    # if beta is just a number, plot a single plot
    else:
        Gain = crystal.small_signal_gain(lambd, beta)
        plt.plot(lambd*1e9, Gain, label=f"$\\beta$ = {beta:.2f}")
    plt.xlabel("wavelength in nm")
    plt.ylabel("Gain G")

    plt.xlim(lam_min,lam_max)
    plt.ylim(bottom=1.1)
    plt.title(f"small signal gain, {crystal.name} at {crystal.temperature}K")
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_small_signal_gain.pdf"))

def plot_beta_eq(crystal):
    plt.figure()
    lambd = crystal.table_sigma_a[:,0]*1e-9
    beta_eq = crystal.beta_eq(lambd)
    params, _ = curve_fit(logistic_function, lambd, beta_eq, p0=[1,980e-9])

    plt.plot(lambd*1e9, beta_eq)
    plt.plot(lambd*1e9, logistic_function(lambd, *params), "--")
    plt.xlabel("wavelength in nm")
    plt.ylabel("equilibrium inversion $\\beta_{eq}$")
    plt.title(f"equilibrium inversion, {crystal.name} at {crystal.temperature}K")
    plt.ylim(-0.1,1.1)

def plot_Isat(crystal):
    plt.figure()
    lambd = crystal.table_sigma_a[:,0]*1e-9

    Isat = h*c/lambd / (crystal.sigma_a(lambd)+crystal.sigma_e(lambd)) / crystal.tau_f * 1e-7

    plt.plot(lambd*1e9, Isat)
    plt.xlabel("wavelength in nm")
    plt.ylabel("saturation intensity")
    plt.title(f"saturation intensity, {crystal.name} at {crystal.temperature}K")
    plt.ylim(0,100)
    plt.xlim(900, 1000)

# =============================================================================
# main script, if this file is executed
# =============================================================================

if __name__ == "__main__":
    # crystal = Crystal(material="YbYAG", temperature=100)
    crystal = Crystal(material="YbCaF2_Toepfer", temperature=300)
    print(crystal)
    plot_beta_eq(crystal)
    plot_Isat(crystal)

    plot_cross_sections(crystal, save=False)
    plot_small_signal_gain(crystal, [0.2,0.22,0.24,0.26,0.28,0.3,0.32], save=False)

    # crystal.smooth_cross_sections(0.7, 1, lambda_min=1050e-9)
    # plot_small_signal_gain(crystal, [0.3,0.22,0.24])
