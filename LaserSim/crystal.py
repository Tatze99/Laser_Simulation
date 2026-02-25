from LaserSim.utilities import numres, h, c, moving_average, fourier_filter, set_plot_params, plot_function, create_save_path, PLOT_DEFAULTS, UNIT_TABLE
import json
import numpy as np
import os
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
set_plot_params()
import matplotlib.pyplot as plt

kB = 1.380649e-23  # Boltzmann constant in J/K
Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

def logistic_function(x, a,b,d):
    return 1/(1+a*np.exp((1/d-1/x)/b*h*c/kB))

class Crystal():
    def __init__(self, material="YbCaF2", temperature=300, lambda_a = None, lambda_e = None, length=None, N_dop=None, tau_f=None, smooth_sigmas = True, useMcCumber_absorption=True, resolution=numres, point_density_reduction=1):
        self.inversion = np.zeros(resolution)
        self.temperature = int(temperature)
        self.use_spline_interpolation = True
        self.material = material
        self.point_density_reduction = point_density_reduction  # reduce the number of points in the cross section data by this (integer) factor

        basedata_path = os.path.join(Folder,"material_database", material, "basedata.json")
        
        self.load_basedata(basedata_path)
        self.load_cross_sections(material)
        self.load_spline_interpolation()

        # overwrite length or doping concentration if given in the constructor
        if lambda_a is not None: self.lambda_a = lambda_a  # absorption wavelength in nm (used for displaying cross sections with print)
        if lambda_e is not None: self.lambda_e = lambda_e  # emission wavelength in nm (used for displaying cross sections with print)
        if tau_f is not None: self.tau_f = tau_f
        if length is not None: self.length = length
        if N_dop is not None: self.doping_concentration = N_dop
        if smooth_sigmas: self.smooth_cross_sections(lambda_max = 1010e-9, useMcCumber=useMcCumber_absorption)

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
            self.lambda_a = self.dict.get("lambda_a", 940)
            self.lambda_e = self.dict.get("lambda_e", 1030)

            unit_key = self.dict.get("wavelength_unit", "nm")
            unit = UNIT_TABLE[unit_key]

            self.wavelength_unit = unit_key
            self.lambda_scale = unit["scale_to_m"]
            self.lambda_label = unit["label"]

            self.energy_lower_level = self.dict.get("energy_lower_level", 0)
            self.energy_upper_level = self.dict.get("energy_upper_level", 1/self.lambda_ZPL*1e-2)
            

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
            self.table_sigma_a = np.nan_to_num(np.loadtxt(sigma_a_path[0]))[::self.point_density_reduction,:]
            self.table_sigma_e = np.nan_to_num(np.loadtxt(sigma_e_path[0]))[::self.point_density_reduction,:]

            if self.table_sigma_a[1,0] < self.table_sigma_a[0,0]: self.table_sigma_a = np.flipud(self.table_sigma_a)
            if self.table_sigma_e[1,0] < self.table_sigma_e[0,0]: self.table_sigma_e = np.flipud(self.table_sigma_e)

            self.table_sigma_a_SI = self.table_sigma_a.copy()
            self.table_sigma_e_SI = self.table_sigma_e.copy()

            self.table_sigma_a_SI[:,0] *= self.lambda_scale  # convert to m
            self.table_sigma_e_SI[:,0] *= self.lambda_scale  # convert to m
            self.table_sigma_a_SI[:,1] *= 1e-4  # convert to m²
            self.table_sigma_e_SI[:,1] *= 1e-4  # convert to m²
            
        else:
            raise FileNotFoundError(f"No file found for file pattern: {Folder} -> material_database -> {self.material} @ {self.temperature}K")

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
        Input: SI unit (meters)
        Returns: absorption cross section at a given wavelength in m²
        Use either the spline interpolation or a linear interpolation
        """
        if self.use_spline_interpolation:
            return self.spline_sigma_a(self.to_display_lambda(lambd))*1e-4
        else:
            return np.interp(self.to_display_lambda(lambd), self.table_sigma_a[:,0], self.table_sigma_a[:,1])*1e-4
        
    def sigma_e(self,lambd):
        """
        Input: SI unit (meters)
        Returns: emission cross section at a given wavelength in m²
        Use either the spline interpolation or a linear interpolation
        """
        if self.use_spline_interpolation:
            return self.spline_sigma_e(self.to_display_lambda(lambd))*1e-4
        else:
            return np.interp(self.to_display_lambda(lambd), self.table_sigma_e[:,0], self.table_sigma_e[:,1])*1e-4
    
    # equilibrium inversion
    def beta_eq(self,lambd):
        """
        Input: wavelength in SI unit (meters)
        Returns: equilibrium inversion at a given wavelength (unitless)
        """
        return self.sigma_a(lambd)/(self.sigma_a(lambd)+self.sigma_e(lambd))

    # saturation fluence
    def F_sat(self,lambd):
        """
        Input: wavelength in SI unit (meters)
        Returns: saturation fluence at a given wavelength in J/m²
        """
        return h*c/lambd /(self.sigma_a(lambd)+self.sigma_e(lambd))
    
    # saturation intensity
    def I_sat(self,lambd):
        """
        Input: wavelength in SI unit (meters)
        Returns: saturation intensity at a given wavelength in W/m²
        """
        return self.F_sat(lambd) / self.tau_f
    
    # Absorption coefficient
    def alpha(self,lambd):
        """
        Input: wavelength in SI unit (meters)
        Returns: absorption coefficient at a given wavelength in 1/m
        """
        return self.doping_concentration * self.sigma_a(lambd)
    
    # Calculate the small signal gain
    def small_signal_gain(self, lambd, beta):
        """
        Input: wavelength in SI unit (meters), inversion beta (unitless)
        Returns: small signal gain G0
        """
        Gain = np.exp(self.doping_concentration*(self.sigma_e(lambd)*beta-self.sigma_a(lambd)*(1-beta))*self.length)
        return Gain
    
    def McCumber_absorption(self, lambda_max=None):
        """
        Use the McCumber relation to approximate the absorption cross section at wavelengths longer than self.lambda_ZPL
        """
        lambd = self.to_internal_lambda(self.table_sigma_a[:,0])
        min_lambda = self.lambda_ZPL + 20e-9
        min_index = np.argmin(np.abs(lambd-min_lambda))
        max_index = np.argmin(np.abs(lambd-lambda_max)) if lambda_max else len(lambd)
        beta_eq = self.beta_eq(lambd)
        try:
            params = fit_beta_eq(self, lambd[min_index:max_index], beta_eq[min_index:max_index], lambda_max=lambda_max)
        except:
            print("Could not fit logistic function to equilibrium inversion to calculate McCumber for absorption using no lambda_max instead")
            params = fit_beta_eq(self, lambd, beta_eq, lambda_max=None)
        interpolated_sigma_e = np.interp(self.to_display_lambda(lambd), self.table_sigma_e[:,0], self.table_sigma_e[:,1])
        self.table_sigma_a[min_index:,1] = interpolated_sigma_e[min_index:] * logistic_function(lambd[min_index:], *params)/(1-logistic_function(lambd[min_index:], *params))

        self.load_spline_interpolation()

    def to_internal_lambda(self, value):
        """User → meters"""
        return value * self.lambda_scale

    def to_display_lambda(self, value):
        """meters → user unit"""
        return value / self.lambda_scale


    def __repr__(self):
        return (
        f"Crystal:\n"
        f"- material = {self.name}\n"
        f"- length = {self.length*1e3} mm\n"
        f"- tau_f = {self.tau_f*1e3} ms\n"
        f"- N_dop = {self.doping_concentration*1e-6} cm^-3\n"
        f"- T = {self.temperature} K\n"
        f"- sigma_a({self.lambda_a}{self.lambda_label}) = {self.sigma_a(self.lambda_a*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_e({self.lambda_a}{self.lambda_label}) = {self.sigma_e(self.lambda_a*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_a({self.lambda_e}{self.lambda_label}) = {self.sigma_a(self.lambda_e*1e-9)*1e4:.3e}cm²\n"
        f"- sigma_e({self.lambda_e}{self.lambda_label}) = {self.sigma_e(self.lambda_e*1e-9)*1e4:.3e}cm²\n\n"
    )

# =============================================================================
# Display of results
# =============================================================================
def fit_beta_eq(crystal, lambd, beta_eq, lambda_max=None):
    """
    Fit the equilibrium inversion with a logistic function.
    """
    index_max = np.argmin(np.abs(lambd - lambda_max)) if lambda_max else len(lambd)
    params, _ = curve_fit(logistic_function, lambd[:index_max], beta_eq[:index_max], 
                          p0=[1,crystal.temperature, crystal.lambda_ZPL], 
                          bounds=([0,crystal.temperature-5,crystal.lambda_ZPL-2e-9],
                                  [np.inf,crystal.temperature+5,crystal.lambda_ZPL+2e-9]))
    
    return params

def plot_cross_sections(crystal, lambda_p=None, lambda_l=None, axis=None, save=False, save_path=None, save_data=False, show_title=True, kwargs=dict(), **options):
    """
    Plot absorption and emission cross sections.
    """
    x = [crystal.table_sigma_a[:, 0], crystal.table_sigma_e[:, 0]]
    y = [crystal.table_sigma_a[:, 1], crystal.table_sigma_e[:, 1]]

    xlabel = f"wavelength in {crystal.lambda_label}"
    ylabel = "cross section in cm²"
    legends = ["absorption $\\sigma_a$", "emission $\\sigma_e$"]
    title = f"{crystal.name} at {crystal.temperature}K" if show_title else None
    fname = f"{crystal.material}_{crystal.temperature}K_cross_sections.pdf"
    path = create_save_path(save_path, fname)

    # Helper function to append sigma values for one wavelength
    def append_sigma_points(x, y, legends, lam, color="tab:green"):
        sigma_a = y[0][np.argmin(abs(lam - x[0]))]
        sigma_e = y[1][np.argmin(abs(lam - x[1]))]

        x.extend([[lam], [lam]])
        y.extend([[sigma_a], [sigma_e]])
        legends.extend([
            f"$σ_a$({lam:g}{crystal.lambda_label}) = {sigma_a:.2e}cm²",
            f"$σ_e$({lam:g}{crystal.lambda_label}) = {sigma_e:.2e}cm²"
        ])

        return [
            dict(marker='o', c=color),
            dict(marker='o', c=color)
        ]

    # now use lambda_p and lambda_l separately
    if not isinstance(kwargs, list): 
        kwargs = [kwargs, kwargs]  # base kwargs for your two original curves

    if lambda_p is not None:
        kwargs.extend(append_sigma_points(x, y, legends, lambda_p, color="tab:green"))

    if lambda_l is not None:
        kwargs.extend(append_sigma_points(x, y, legends, lambda_l, color="tab:red"))

    plot_function(x, y, xlabel, ylabel, title, legends, axis, save, path, save_data, kwargs=kwargs)

def plot_small_signal_gain(crystal, beta, round_trips=1, normalize=False, xlim=None, ylim=(1.1, np.inf), save=False, save_path=None, save_data=False, show_title=True, axis=None, double_pass=True):
    """
    Plot small signal gain for a given beta.
    """
    if xlim is None:
        bandwidth = crystal.to_display_lambda(30e-9)  # 30 nm bandwidth around the emission wavelength
        xlim = (crystal.lambda_e-bandwidth, crystal.lambda_e+bandwidth)

    lambd = crystal.to_internal_lambda(np.linspace(xlim[0], xlim[1], 100))
    if double_pass:
        factor = 2
    else:
        factor = 1
    
    if not isinstance(beta, (list, tuple, np.ndarray)):
        beta = [beta]
        
    xlabel = f"wavelength in {crystal.lambda_label}"
    ylabel = "Gain G"
    title = f"small signal gain, {crystal.name} at {crystal.temperature}K" if show_title else None
    y_list = [crystal.small_signal_gain(lambd, b)**(factor*round_trips) for b in beta]
    if normalize:
        y_list = [y/np.max(y) for y in y_list]
        ylim = (0,1.1)
        ylabel = "normalized Gain G"

    if round_trips != 1:
        title += f", for {round_trips} round trips"

    legends = [f"$\\beta$ = {b:.2f}" for b in beta]
    beta_name = "_".join(str(round(b,2)) for b in beta)
    RT_name = f"_{round_trips}RT" if round_trips != 1 else ""

    fname = f"{crystal.material}_{crystal.temperature}K_small-signal-gain_beta_{beta_name}{RT_name}.pdf"
    path = create_save_path(save_path, fname)

    plot_function(crystal.to_display_lambda(lambd), y_list, xlabel, ylabel, title, legends, axis, save, path, save_data, xlim=xlim, ylim=ylim)

def plot_beta_eq(crystal, lambda_p=None, lambda_l=None, lambda_max=None, save=False, save_path=None, save_data=False, show_title=True, kwargs=dict(),axis=None):
    """
    Plot equilibrium inversion beta_eq with a logistic fit.
    """
    lambd = crystal.table_sigma_a_SI[:, 0]
    beta_eq = crystal.beta_eq(lambd)
    params = fit_beta_eq(crystal, lambd, beta_eq, lambda_max=lambda_max)
    
    y_list = [beta_eq, logistic_function(lambd, *params)]
    xlabel = f"wavelength in {crystal.lambda_label}"
    ylabel = "equilibrium inversion $\\beta_{eq}$"
    title = f"equilibrium inversion, {crystal.name} at {crystal.temperature}K" if show_title else None
    legends = ["Beta Eq", f"Fit, Zl/Zu = {params[0]:.2f}, T = {params[1]:.1f}K, ZPL = {crystal.to_display_lambda(params[2]):.1f}{crystal.lambda_label}"]

    fname = f"{crystal.material}_{crystal.temperature}K_equilibrium_inversion.pdf"
    path = create_save_path(save_path, fname)

    # now use lambda_p and lambda_l separately
    if not isinstance(kwargs, list): 
        kwargs = [kwargs, kwargs]  # base kwargs for your two original curves
    
    x = [crystal.to_display_lambda(lambd), crystal.to_display_lambda(lambd)]

    # Helper function to append sigma values for one wavelength
    def append_sigma_points(x, y, legends, lam, color="tab:green"):
        beta = y[1][np.argmin(abs(lam - x[1]))]

        x.extend([lam])
        y.extend([beta])
        legends.extend([
            f"$\\beta_{{eq}}$({lam:g}{crystal.lambda_label}) = {beta:.2f}"
        ])

        return [dict(marker='o', c=color)]

    if lambda_p is not None:
        kwargs.extend(append_sigma_points(x, y_list, legends, lambda_p, color="tab:green"))

    if lambda_l is not None:
        kwargs.extend(append_sigma_points(x, y_list, legends, lambda_l, color="tab:red"))

    plot_function(x, y_list, xlabel, ylabel, title, legends, axis, save, path, save_data, ylim=(-.1,1.1), kwargs=kwargs)

def plot_Isat(crystal, save=False, save_path=None, save_data=False, xlim=None, ylim=(0,200), lambda0 = None, legends=None, show_title=True, axis=None, kwargs=None):
    """
    Plot the saturation intensity of a crystal.
    """
    if xlim is None:
        bandwidth = crystal.to_display_lambda(50e-9)  # 50 nm bandwidth around the emission wavelength
        xlim = (crystal.lambda_a-bandwidth, crystal.lambda_a+bandwidth)
    lambd = crystal.table_sigma_a_SI[:, 0]
    Isat = crystal.I_sat(lambd) * 1e-7
    xlabel = f"wavelength in {crystal.lambda_label}"
    ylabel = "$I_{sat}$ in kW/cm²"
    title = f"saturation intensity, {crystal.name} at {crystal.temperature}K" if show_title else None
    fname = f"{crystal.material}_{crystal.temperature}K_Isat.pdf"
    path = create_save_path(save_path, fname)
    lambd = crystal.to_display_lambda(lambd)

    if lambda0 is not None:
        Isat_lambda0 = crystal.I_sat(crystal.to_internal_lambda(lambda0)) * 1e-7
        lambd = [lambd, lambda0]
        Isat = [Isat, Isat_lambda0]
        legends = ["saturation intensity", f"Isat = {Isat_lambda0:.2f}kW/cm² at {lambda0:g} {crystal.lambda_label}"]
        kwargs=[dict(),dict(marker='o')]

    plot_function(lambd, Isat, xlabel, ylabel, title, legends, axis, save, path, save_data, xlim=xlim, ylim=ylim, kwargs=kwargs)

def plot_Fsat(crystal, save=False, save_path=None, save_data=False, xlim=None, ylim=(0,200), lambda0 = None, legends=None, show_title=True, axis=None, kwargs=None):
    """
    Plot the saturation fluence of a crystal.
    """
    if xlim is None:
        bandwidth = crystal.to_display_lambda(30e-9)  # 30 nm bandwidth around the emission wavelength
        xlim = (crystal.lambda_e-bandwidth, crystal.lambda_e+bandwidth)
    lambd = crystal.table_sigma_a_SI[:, 0]
    Fsat = crystal.F_sat(lambd) * 1e-4
    xlabel = f"wavelength in {crystal.lambda_label}"
    ylabel = "$F_{sat}$ in J/cm²"
    title = f"saturation fluence, {crystal.name} at {crystal.temperature}K" if show_title else None
    fname = f"{crystal.material}_{crystal.temperature}K_Fsat.pdf"
    path = create_save_path(save_path, fname)
    lambd = crystal.to_display_lambda(lambd)

    if lambda0 is not None:
        Fsat_lambda0 = crystal.F_sat(crystal.to_internal_lambda(lambda0)) * 1e-4
        lambd = [lambd, lambda0]
        Fsat = [Fsat, Fsat_lambda0]
        legends = ["saturation fluence", f"Fsat = {Fsat_lambda0:.2f}J/cm² at {lambda0:g} {crystal.lambda_label}"]
        kwargs=[dict(),dict(marker='o')]

    plot_function(lambd, Fsat, xlabel, ylabel, title, legends, axis, save, path, save_data, xlim=xlim, ylim=ylim, kwargs=kwargs)

def plot_lambert_beer(crystal, lambda0 = None, axis=None, save=False, save_data=False, save_path=None, show_title=True, custom_legend=""):
    """
    Plot the absorbed pump energy as a fraction of the total input fluence 
    """
    lambda0 = lambda0 if lambda0 else crystal.lambda_a
    x = crystal.z_axis*1e3
    y = np.exp(-crystal.alpha(crystal.to_internal_lambda(lambda0))*crystal.z_axis)
    print("lambda, alpha", crystal.to_internal_lambda(lambda0), crystal.alpha(crystal.to_internal_lambda(lambda0)))

    xlabel = "depth $z$ in mm"
    ylabel = "transmitted light fraction $I(z)/I_0$"
    title = "transmitted light with Lambert Beer's law" if show_title else None
    legend = [f"$T$ ({x[-1]}mm) = {np.min(y)*1e2:.1f} % at {lambda0:g} {crystal.lambda_label}"] if custom_legend == "" else custom_legend
    fname = f"{crystal.material}_{crystal.temperature}K_Lambert_Beer_transmission_vs_crystal_thickness.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data)

#=============================================================================
# main script, if this file is executed
# ============================================================================

if __name__ == "__main__":
    crystal = Crystal(material="YbCaF2", temperature=300, smooth_sigmas=True)
    pump_wavelength = crystal.lambda_a
    laser_wavelength = crystal.lambda_e

    print(crystal)
    plot_cross_sections(crystal, save=False, lambda_p=pump_wavelength, lambda_l=laser_wavelength)
    plot_beta_eq(crystal, save=False)
    plot_Isat(crystal, save=False, lambda0=pump_wavelength)
    plot_Fsat(crystal, save=False, lambda0=laser_wavelength)

    plot_small_signal_gain(crystal, [0.16,0.18,0.2,0.22,0.24,0.26,0.28], round_trips=1, normalize=False, save=False)