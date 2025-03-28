from LaserSim.utilities import numres, h, c, moving_average, fourier_filter, set_plot_params, plot_function
import json
import numpy as np
import os
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
set_plot_params()

Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

def logistic_function(x, a,b):
    return 1/(1+np.exp(-a*(x-b)))

class Crystal():
    def __init__(self, material="YbCaF2", temperature="300", lambda_a = 940, lambda_e = 1030, length=None, N_dop=None, smooth_sigmas = True, resolution=numres):
        self.inversion = np.zeros(resolution)
        self.temperature = temperature
        self.use_spline_interpolation = True
        self.material = material
        self.lambda_a = lambda_a  # absorption wavelength in nm (used for displaying cross sections with print)
        self.lambda_e = lambda_e  # emission wavelength in nm (used for displaying cross sections with print)    

        basedata_path = os.path.join(Folder,"material_database", material, "basedata.json")
        
        self.load_basedata(basedata_path)
        self.load_cross_sections(material)
        self.load_spline_interpolation()

        # overwrite length or doping concentration if given in the constructor
        if length is not None: self.length = length
        if N_dop is not None: self.doping_concentration = N_dop
        if smooth_sigmas: self.smooth_cross_sections(lambda_max = 1010e-9)

        self.z_axis = np.linspace(0, self.length, resolution)
        self.dz = self.length / (resolution-1)
        self.inversion_end = None

    def load_basedata(self, filename):
        """
        load the basic crystal data from a json file
        """
        with open(filename, "r") as file:
            self.dict = json.load(file)
            self.length = self.dict["length"]
            self.tau_f = self.dict["tau_f"]
            self.doping_concentration = self.dict["N_dop"]
            self.name = self.dict["name"]
            self.lambda_ZPL = self.dict["ZPL"]

    def load_spline_interpolation(self):
        """"
        Perform a spline interpolation of the absorption and emission cross sections
        """
        self.spline_sigma_a = CubicSpline(self.table_sigma_a[:,0], self.table_sigma_a[:,1], bc_type="clamped")
        self.spline_sigma_e = CubicSpline(self.table_sigma_e[:,0], self.table_sigma_e[:,1], bc_type="clamped")

    def load_cross_sections(self, material):
        """
        load the absorption and emission cross sections from the database
        """
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

    def smooth_cross_sections(self, FF_filter=0, mov_average=4, useMcCumber = True, lambda_max=None):
        """
        smooth the cross sections with a fourier filter and a moving average    
        FF_filter: fourier filter cutoff: 0 ... 1 (0: no filter, 1: all frequencies are filtered)
        mov_average: number of neighbouring points for the moving average        
        """
        self.table_sigma_a[:,1] = fourier_filter(self.table_sigma_a, FF_filter)
        self.table_sigma_e[:,1] = fourier_filter(self.table_sigma_e, FF_filter)
        self.table_sigma_a[:,1] = moving_average(self.table_sigma_a[:,1], mov_average)
        self.table_sigma_e[:,1] = moving_average(self.table_sigma_e[:,1], mov_average)
        if useMcCumber:
            self.McCumber_absorption(lambda_max=lambda_max)
        self.load_spline_interpolation()
        
    def sigma_a(self,lambd):
        """
        return the absorption cross section at a given wavelength in m²
        Use either the spline interpolation or a linear interpolation
        """
        if self.use_spline_interpolation:
            return self.spline_sigma_a(lambd*1e9)*1e-4
        else:
            return np.interp(lambd*1e9, self.table_sigma_a[:,0], self.table_sigma_a[:,1])*1e-4
        
    def sigma_e(self,lambd):
        """
        return the emission cross section at a given wavelength in m²
        Use either the spline interpolation or a linear interpolation
        """
        if self.use_spline_interpolation:
            return self.spline_sigma_e(lambd*1e9)*1e-4
        else:
            return np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])*1e-4
    
    # equilibrium inversion
    def beta_eq(self,lambd):
        return self.sigma_a(lambd)/(self.sigma_a(lambd)+self.sigma_e(lambd))

    # saturation fluence
    def F_sat(self,lambd):
        return h*c/lambd /(self.sigma_a(lambd)+self.sigma_e(lambd))
    
    # saturation intensity
    def I_sat(self,lambd):
        return self.F_sat(lambd) / self.tau_f
    
    # Absorption coefficient
    def alpha(self,lambd):
        return self.doping_concentration * self.sigma_a(lambd)
    
    # Calculate the small signal gain
    def small_signal_gain(self, lambd, beta):
        Gain = np.exp(self.doping_concentration*(self.sigma_e(lambd)*beta-self.sigma_a(lambd)*(1-beta))*self.length)
        return Gain
    
    def McCumber_absorption(self, lambda_max=None):
        """
        Use the McCumber relation to approximate the absorption cross section at wavelengths longer than self.lambda_ZPL
        """
        lambd = self.table_sigma_a[:,0]*1e-9
        min_index = np.argmin(np.abs(lambd-self.lambda_ZPL-20e-9))
        max_index = np.argmin(np.abs(lambd-lambda_max)) if lambda_max else len(lambd)
        beta_eq = self.beta_eq(lambd)
        print(lambda_max, self.lambda_ZPL, min_index, max_index)
        try:
            params, _ = curve_fit(logistic_function, lambd[:max_index], beta_eq[:max_index], p0=[1,self.lambda_ZPL])
        except:
            print("Could not fit logistic function to equilibrium inversion to calculate McCumber for absorption using no lambda_max instead")
            params, _ = curve_fit(logistic_function, lambd, beta_eq, p0=[1,self.lambda_ZPL])
        interpolated_sigma_e = np.interp(lambd*1e9, self.table_sigma_e[:,0], self.table_sigma_e[:,1])
        self.table_sigma_a[min_index:,1] = interpolated_sigma_e[min_index:] * logistic_function(lambd[min_index:], *params)/(1-logistic_function(lambd[min_index:], *params))
        self.load_spline_interpolation()

    def __repr__(self):
        return (
        f"Crystal:\n"
        f"- material = {self.name}\n"
        f"- length = {self.length*1e3} mm\n"
        f"- tau_f = {self.tau_f*1e3} ms\n"
        f"- N_dop = {self.doping_concentration*1e-6} cm^-3\n"
        f"- sigma_a({self.lambda_a}nm) = {self.sigma_a(self.lambda_a*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_e({self.lambda_a}nm) = {self.sigma_e(self.lambda_a*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_a({self.lambda_e}nm) = {self.sigma_a(self.lambda_e*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_e({self.lambda_e}nm) = {self.sigma_e(self.lambda_e*1e-9)*1e4:.3e}cm²\n\n"
    )

# =============================================================================
# Display of results
# =============================================================================

def plot_cross_sections(crystal, save=False, save_path=None):
    """
    Plot absorption and emission cross sections.
    """
    x = [crystal.table_sigma_a[:, 0], crystal.table_sigma_e[:, 0]]
    y = [crystal.table_sigma_a[:, 1], crystal.table_sigma_e[:, 1]]
    xlabel = "wavelength in nm"
    ylabel = "cross section in cm²"
    legends = ["absorption $\\sigma_a$", "emission $\\sigma_e$"]
    title = f"{crystal.name} at {crystal.temperature}K"
    path = save_path or os.path.join(Folder, "material_database", "plots", f"{crystal.material}_{crystal.temperature}K_cross_sections.pdf")
    
    plot_function(x, y, xlabel, ylabel, title, legends, save, path)


def plot_small_signal_gain(crystal, beta, xlim=(1000,1060), ylim=(1.1, np.inf), save=False, save_path=None):
    """
    Plot small signal gain for a given beta.
    """
    lambd = np.linspace(xlim[0] * 1e-9, xlim[1] * 1e-9, 100)
    
    if not isinstance(beta, (list, tuple, np.ndarray)):
        beta = [beta]
        
    y_list = [crystal.small_signal_gain(lambd, b)**2 * (1 - 0.078) for b in beta]
    xlabel = "wavelength in nm"
    ylabel = "Gain G"
    title = f"small signal gain, {crystal.name} at {crystal.temperature}K"
    legends = [f"$\\beta$ = {b:.2f}" for b in beta]

    path = save_path or os.path.join(Folder, "material_database", "plots", 
            f"{crystal.material}_{crystal.temperature}K_small_signal_gain.pdf")

    print(path)
    plot_function(lambd * 1e9, y_list, xlabel, ylabel, title, legends, save, path, xlim=xlim, ylim=ylim)


def plot_beta_eq(crystal, lambda_max=None, save=False, save_path=None):
    """
    Plot equilibrium inversion beta_eq with a logistic fit.
    """
    lambd = crystal.table_sigma_a[:, 0] * 1e-9
    beta_eq = crystal.beta_eq(lambd)
    index_max = np.argmin(np.abs(lambd - lambda_max)) if lambda_max else len(lambd)
    params, _ = curve_fit(logistic_function, lambd[:index_max], beta_eq[:index_max], p0=[1, crystal.lambda_ZPL])
    
    y_list = [beta_eq, logistic_function(lambd, *params)]
    xlabel = "wavelength in nm"
    ylabel = "equilibrium inversion $\\beta_{eq}$"
    title = f"equilibrium inversion, {crystal.name} at {crystal.temperature}K"
    legends = ["Beta Eq", "Fit"]
    path = save_path or os.path.join(Folder, "material_database", "plots",
            f"{crystal.material}_{crystal.temperature}K_equilibrium_inversion.pdf")

    plot_function(lambd * 1e9, y_list, xlabel, ylabel, title, legends, save, path, ylim=(-.1,1.1))


def plot_Isat(crystal, save=False, save_path=None, xlim=(900,1000), ylim=(0,200)):
    """
    Plot the saturation intensity of a crystal.
    """
    lambd = crystal.table_sigma_a[:, 0] * 1e-9
    Isat = crystal.I_sat(lambd) * 1e-7
    xlabel = "wavelength in nm"
    ylabel = "$I_{sat}$ in kW/cm²"
    title = f"saturation intensity, {crystal.name} at {crystal.temperature}K"
    path = save_path or os.path.join(Folder, "material_database", "plots",f"{crystal.material}_{crystal.temperature}K_Isat.pdf")

    plot_function(lambd * 1e9, Isat, xlabel, ylabel, title, save=save, save_path=path, xlim=xlim, ylim=ylim)

def plot_Fsat(crystal, save=False, save_path=None, xlim=(1010,1050), ylim=(0,200)):
    """
    Plot the saturation fluence of a crystal.
    """
    lambd = crystal.table_sigma_a[:, 0] * 1e-9
    Isat = crystal.F_sat(lambd) * 1e-4
    xlabel = "wavelength in nm"
    ylabel = "$F_{sat}$ in J/cm²"
    title = f"saturation fluence, {crystal.name} at {crystal.temperature}K"
    path = save_path or os.path.join(Folder, "material_database", "plots",f"{crystal.material}_{crystal.temperature}K_Isat.pdf")

    plot_function(lambd * 1e9, Isat, xlabel, ylabel, title, save=save, save_path=path, xlim=xlim, ylim=ylim)
    
#=============================================================================
# main script, if this file is executed
# ============================================================================

if __name__ == "__main__":
    crystal = Crystal(material="YbLiAS15", temperature=300, smooth_sigmas=True)

    print(crystal)
    plot_cross_sections(crystal, save=False)
    plot_beta_eq(crystal, save=False)
    plot_Isat(crystal, save=False)

    plot_small_signal_gain(crystal, [0.2,0.22,0.24,0.26,0.28,0.3,0.32], save=False)

