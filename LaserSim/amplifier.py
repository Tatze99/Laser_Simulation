from LaserSim.crystal import Crystal, plot_beta_eq
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.spectral_losses import Spectral_Losses
from LaserSim.utilities import z_integ, t_integ, integ, numres, h, c, set_plot_params, plot_function, create_save_path

from matplotlib.transforms import offset_copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
import os
import copy

set_plot_params()
Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

class Amplifier():

    def __init__(self, crystal=Crystal(), pump=Pump(), seed=Seed(), passes = 50, losses = 2e-2, print_iteration = False, spectral_losses = None, max_fluence=10, fast_CPA_computing=False):
        self.pump = pump
        self.seed = seed
        self.crystal = crystal
        self.passes = passes
        self.losses = losses
        self.print_iteration = print_iteration
        self.spectral_losses = spectral_losses
        self.max_fluence = max_fluence*1e4  # [J/m²]
        self.fast_CPA_computing = fast_CPA_computing  # if True, the CPA calculation is faster, but doesn't include spectral edge steepening, i.e. reduction of the inversion is calculated once at the end of the pass (using a mean saturation and Gain value) and not after every wavelength has passed. This is only ,valid when saturation is low.

    def averaged_beta_eq(self):
        return integ(self.crystal.beta_eq(self.pump.lambdas)*self.pump.pulse, self.pump.dlambda)[-1] / integ(self.pump.pulse, self.pump.dlambda)[-1]
    
    def inversion(self, pump_intensity = None):

        if pump_intensity is None:
            pump_intensity = self.pump.intensity

        if self.pump.bandwidth > 0:
            beta_eq = self.averaged_beta_eq()
        else:
            beta_eq = self.crystal.beta_eq(self.pump.wavelength)

        self.pump_inversion = beta_eq
        tau_f = self.crystal.tau_f
        sigma_a = self.crystal.sigma_a(self.pump.wavelength)
        alpha = self.crystal.doping_concentration * sigma_a
        beta_zero = 0
        z_axis = self.crystal.z_axis
        t_axis = self.pump.t_axis
        

        def beta(pump_rate, beta_zero=beta_zero, beta_eq=beta_eq, tau_f=tau_f, dt=self.pump.dt):
            exponent = np.exp(t_integ(pump_rate / beta_eq + 1/tau_f, dt) )
            return 1/exponent * (t_integ(pump_rate*exponent, dt ) + beta_zero)

        def Rb(beta, alpha = alpha, beta_eq = beta_eq, dz = self.crystal.dz):
            return np.exp(z_integ(beta * alpha/beta_eq, dz))

        def break_condition(beta_low, beta_high, it):
            max_it = 30
            abweichung = np.abs(np.max(beta_high-beta_low))
            fehler = 1/np.max([numres*t_axis[-1]/tau_f, 1e3])

            A = (it > max_it)
            B = (abweichung < fehler)
            if A and self.print_iteration:
                print("Max iterations of " + str(it) + " exceeded.")
            if B and self.print_iteration:
                print("Deviation between beta_low and beta_high =",
                    abweichung, "<", fehler)
            return A or B

        # Initialisierung
        beta_low = np.zeros((len(t_axis), len(z_axis)))
        beta_high = np.ones_like(beta_low) * beta_eq
        R0 = pump_intensity * sigma_a / h / (c / self.pump.wavelength)
        lambert_beer = np.exp(-alpha * z_axis)
        
        Ra = R0 * npm.repmat(lambert_beer, len(t_axis), 1)
        iteration = 0
        betas = []
        pumprates = []

        # Iteration
        while not break_condition(beta_low, beta_high, iteration):
            iteration += 1
            if self.print_iteration: print("Iteration number: " + str(iteration))

            if self.pump.symmetric:
                R_high = (Ra * Rb(beta_high) + np.flip(Ra, axis=1) * np.flip(Rb(np.flip(beta_high, axis=1)), axis=1))/2
                R_low = (Ra * Rb(beta_low) + np.flip(Ra, axis=1) *np.flip(Rb(np.flip(beta_low, axis=1)), axis=1))/2
            else:
                R_high = Ra * Rb(beta_high)
                R_low = Ra * Rb(beta_low)

            beta_high = beta(R_high)
            beta_low = beta(R_low)

            betas.append((beta_low, beta_high))
            pumprates.append((R_low, R_high))
    
        beta_total = (beta_low + beta_high) / 2
        pumprate = (R_high + R_low) / 2

        stored_fluence = h * c / self.seed.wavelength * self.crystal.doping_concentration * np.sum(beta_total - self.crystal.beta_eq(self.seed.wavelength), 1) * self.crystal.dz
        absorbed_energy = h * c / self.pump.wavelength *  self.crystal.doping_concentration * integ(beta_total[-1,:], self.crystal.dz) / (self.pump.intensity * self.pump.duration)

        self.crystal.inversion = beta_total
        self.pump.pumprate = pumprate
        self.pump.absorbed_energy = absorbed_energy
        self.crystal.inversion_end = beta_total[-1,:]
        self.crystal.stored_fluence = stored_fluence
        self.beta_0 = np.mean(self.crystal.inversion_end)

        return beta_total[-1,:]

    # =============================================================================
    # Calculate energy extraction for a monochromatic, temporal pulse
    # =============================================================================
    
    def extraction(self):
        """
        Calculate the energy extraction for a monochromatic, temporal pulse.
        Returns the total fluence at the end of each pass and the temporal fluence at the end of each pass.
        """
        if np.any(self.crystal.inversion_end):
            beta_0 = self.crystal.inversion_end           
        else:
            beta_0 = self.inversion()

        pulse_out = np.zeros((self.passes+1, len(self.seed.pulse)))
        beta_out  = np.zeros( (self.passes+1, len(beta_0)) )
        pulse_out[0,:] = self.seed.pulse
        beta_out[0,:] = np.abs(beta_0)
        sigma_tot = self.crystal.sigma_a(self.seed.wavelength) + self.crystal.sigma_e(self.seed.wavelength)
        break_index = self.passes+1

        # equilibrium inversion and absorption coefficient at the seed wavelength
        beta_eq = self.crystal.beta_eq(self.seed.wavelength)
        alpha   = self.crystal.alpha(self.seed.wavelength)
        
        for k in range(self.passes):
		# calculate single components
            if k >= 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            # Gain G(z), spatial array of the integrated small signal gain
            Gain = np.exp(-alpha * integ(1-beta_out[k,:]/beta_eq, self.crystal.dz))

            # Saturation S(t), temporal array of the saturation, Saturation_end: S(t=t_end)
            Saturation = np.exp(sigma_tot * c * integ(pulse_out[k, :], self.seed.dt))
    
            # compute temporal pulse shape after amplification at z = length
            pulse_out[k+1,:] = pulse_out[k,:] * (Saturation*Gain[-1])/(1+(Saturation-1)*Gain[-1])

            if k % 2: pulse_out[k+1,:] *= (1-self.losses)

            # compute beta after amplification at t = t_end
            beta_out[k+1,:] = beta_eq + (beta_out[k,:] - beta_eq) / (1 + (Saturation[-1] - 1)*Gain)

            if integ(pulse_out[k+1,:], self.seed.dt)[-1]* c * h *c / self.seed.wavelength > self.max_fluence:
                break_index = k+2
                break

        self.pulse_out = pulse_out[:break_index,:]
        self.temporal_fluence_out = self.pulse_out * c * h *c / self.seed.wavelength 
        self.total_fluence_out = z_integ(self.temporal_fluence_out, self.seed.dt)[:,-1]
        self.max_gain = np.max(self.total_fluence_out[1::] / self.total_fluence_out[0:-1])
        # change direction of the beta_out as it is flipped after every pass
        self.beta_out = beta_out[-1,::-1] if break_index % 2 else beta_out[-1,:]

        return self.temporal_fluence_out

    # =============================================================================
    # Calculate energy extraction for a CPA pulse
    # =============================================================================

    def extraction_CPA_old(self):
        """
        Calculate the energy extraction for a CPA pulse. 
        Returns the spectral fluence at the end of each pass.
        """
        if np.any(self.crystal.inversion_end):
            beta_0 = self.crystal.inversion_end           
        else:
            beta_0 = self.inversion()

        if self.spectral_losses is None:
            self.spectral_losses = np.zeros(len(self.seed.lambdas)) 
        

        beta_out = np.zeros( (self.passes+1, len(beta_0)) )
        beta_out[0,:] = np.abs(beta_0)
        spectral_fluence_out = np.zeros( (self.passes+1, len(self.seed.spectral_fluence)))
        spectral_fluence_out[0] = self.seed.spectral_fluence

        beta_eq = self.crystal.beta_eq(self.seed.lambdas)
        sigma_a = self.crystal.sigma_a(self.seed.lambdas)
        sigma_e = self.crystal.sigma_e(self.seed.lambdas)
        Fsat = self.crystal.F_sat(self.seed.lambdas)
        break_index = self.passes+1

        for k in range(self.passes):
		# calculate single components
            # if k > 1:
            #     beta_out[k, :] = np.flipud(beta_out[k, :])

            # Compute the wavelength dependent saturation
            Saturation = np.exp(integ(spectral_fluence_out[k,:], self.seed.dlambda)[-1]/Fsat)

            # Compute the small signal gain by integrating along z
            Gain = np.exp(self.crystal.doping_concentration*z_integ(np.outer(sigma_e,beta_out[k,:])-np.outer(sigma_a,(1-beta_out[k,:])),self.crystal.dz))[:,-1]

            # Calculate the spectral fluence using Frantz Nodvik
            spectral_fluence_out[k+1, :] = spectral_fluence_out[k, :] * Fsat/(integ(spectral_fluence_out[k,:], self.seed.dlambda)[-1]) * np.log(1+Gain*(Saturation-1))

            if k % 2:
                spectral_fluence_out[k+1, :] *= (1-self.losses-self.spectral_losses)
            beta_out[k+1,:] = np.mean(beta_eq) + (beta_out[k,:]-np.mean(beta_eq))/(1+np.mean(Gain)*(np.mean(Saturation)-1))
            
            beta_out[k,:] = beta_out[k,::-1]

            if integ(spectral_fluence_out[k+1,:], self.seed.dlambda)[-1] > self.max_fluence:
                break_index = k+2
                break
        
        self.total_fluence_out = z_integ(spectral_fluence_out[:break_index,:], self.seed.dlambda)[:,-1]
        self.spectral_fluence_out = spectral_fluence_out[:break_index,:] 
        self.beta_out = beta_out if break_index % 2 else beta_out[::-1]

        return spectral_fluence_out

    # =============================================================================
    # Calculate energy extraction for a CPA pulse
    # =============================================================================

    def extraction_CPA(self):
        """
        Calculate the energy extraction for a CPA pulse. 
        Returns the spectral fluence at the end of each pass.
        """
        if np.any(self.crystal.inversion_end):
            self.beta_0 = self.crystal.inversion_end           
        else:
            self.beta_0 = self.inversion()

        if self.spectral_losses is None:
            self.spectral_losses = np.zeros(len(self.seed.lambdas)) 
        
        beta_out = np.abs(self.beta_0)
        spectral_fluence_out = np.zeros( (self.passes+1, len(self.seed.spectral_fluence)))
        spectral_fluence_out[0] = self.seed.spectral_fluence

        sigma_e_array = self.crystal.sigma_e(self.seed.lambdas)
        sigma_a_array = self.crystal.sigma_a(self.seed.lambdas)
        Fsat_array = self.crystal.F_sat(self.seed.lambdas)
        beta_eq_array = self.crystal.beta_eq(self.seed.lambdas)
        break_index = self.passes+1
        dlambda = self.seed.dlambda

        for k in range(self.passes):

            if self.fast_CPA_computing:
                Saturation = np.exp(integ(spectral_fluence_out[k,:], self.seed.dlambda)[-1]/Fsat_array)
                # Compute the small signal gain by integrating along z
                Gain = np.exp(self.crystal.doping_concentration*z_integ(np.outer(sigma_e_array,beta_out)-np.outer(sigma_a_array,(1-beta_out)),self.crystal.dz))[:,-1]

                # Calculate the spectral fluence using Frantz Nodvik
                spectral_fluence_out[k+1, :] = spectral_fluence_out[k, :] * Fsat_array/(integ(spectral_fluence_out[k,:], dlambda)[-1]) * np.log(1+Gain*(Saturation-1))

                beta_out = np.mean(beta_eq_array) + (beta_out-np.mean(beta_eq_array))/(1+np.mean(Gain)*(np.mean(Saturation)-1))

            else:
                Saturation_array = np.exp(spectral_fluence_out[k,:]*dlambda/Fsat_array)
                iterator = zip(sigma_e_array, sigma_a_array, beta_eq_array, Fsat_array, Saturation_array)

                for n, (sigma_e, sigma_a, beta_eq, Fsat, Saturation) in enumerate(iterator):

                    Gain = np.exp(self.crystal.doping_concentration*integ(sigma_e*beta_out-(1-beta_out)*sigma_a,self.crystal.dz))
                    spectral_fluence_out[k+1, n] = Fsat * np.log(1+Gain[-1]*(Saturation-1))/dlambda
                    
                    beta_out = beta_eq + (beta_out-beta_eq)/(1+Gain*(Saturation-1))
            
            if k % 2: spectral_fluence_out[k+1, :] *= (1-self.losses-self.spectral_losses)

            beta_out = beta_out[::-1]

            if integ(spectral_fluence_out[k+1,:], dlambda)[-1] > self.max_fluence:
                break_index = k+2
                break

        self.spectral_fluence_out = spectral_fluence_out[:break_index,:]
        self.total_fluence_out = z_integ(self.spectral_fluence_out[:break_index,:], self.seed.dlambda)[:,-1]
        self.beta_out = beta_out if break_index % 2 else beta_out[::-1]

        return self.spectral_fluence_out
    
    def storage_efficiency(self, pump_intensity=None, pump_duration=None):
        original_pump = copy.deepcopy(self.pump)
        """
        Calculate the storage efficiency of the crystal.
        """
        if pump_intensity is None: pump_intensity = self.pump.intensity
        if pump_duration is None: pump_duration = self.pump.duration

        self.pump.duration = pump_duration
        self.pump.t_axis = np.linspace(0, pump_duration, numres)
        self.pump.dt = pump_duration / numres

        if not isinstance(pump_intensity, (list, np.ndarray)):
            pump_intensity = np.array([pump_intensity])

        extractable_fluence = np.zeros((len(pump_intensity), len(self.pump.t_axis)))

        for i, I_p in enumerate(pump_intensity):
            _ = self.inversion(pump_intensity=I_p)
            inversion = self.crystal.inversion

            extractable_fluence[i, :] = z_integ(inversion - self.crystal.beta_eq(self.seed.wavelength), self.crystal.dz)[:,-1] * h * c * self.crystal.doping_concentration / (self.seed.wavelength)

        I, T = np.meshgrid(pump_intensity, self.pump.t_axis, indexing="ij")
        with np.errstate(divide='ignore', invalid='ignore'):
            # ignore divide by zero warning for for starting values of the arrays
            efficiency = extractable_fluence / (I * T)
            efficiency = np.where(efficiency < 0, 0, efficiency)

        self.pump = original_pump

        return efficiency
    
    def __repr__(self):
        return (f"{self.crystal}{self.pump}{self.seed}"
                f"Amplifier:\n"
                f"- medium passes = {self.passes}\n"
                f"- losses = {self.losses*1e2}% \n"
                f"- maximum allowed fluence = {self.max_fluence*1e-4}J/cm²\n"
                f"- equilibrium inversion (pump) = {self.crystal.beta_eq(self.pump.wavelength):.5f} @ {self.pump.wavelength*1e9:.2f} nm\n"
                f"- saturation intensity (pump) = {self.crystal.I_sat(self.pump.wavelength)*1e-7:.2f} kW/cm² @ {self.pump.wavelength*1e9:.2f} nm\n"
                f"- saturation fluence (laser) = {self.crystal.F_sat(self.seed.wavelength)*1e-4:.2f} J/cm² @ {self.seed.wavelength*1e9:.2f} nm\n"
        )
# =============================================================================
# Display of results
# =============================================================================

def plot_temporal_fluence(amplifier, axis=None, save=False, save_path=None, save_data=False, show_title=True, normalize=False):
    """ 
    Plot the temporal fluence for the last ten passes
    """
    pulse_out = amplifier.extraction()
    passes = len(pulse_out[:,0])-1

    x = amplifier.seed.time*1e9
    y_array = []
    legend = []
    for i in range(max(passes-20,0), passes+1, 2):
        y_array.append(pulse_out[i,:]*1e-9*1e-4)
        legend.append(f"$F =${integ(pulse_out[i,:], amplifier.seed.dt)[-1]*1e-4:.3f}J/cm²")

    xlabel="time $t$ in s"
    ylabel="temporal fluence in J/cm²/ns"
    title = f"{amplifier.crystal.name} with {int(passes/2)} RT, F$_0$ = {amplifier.seed.fluence*1e-4:.2e} J/cm²" if show_title else None

    fname = f"Temporal_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}_temporal-fluence.pdf"
    path = create_save_path(save_path, fname)

    kwargs = dict(marker="o")

    plot_function(x,y_array, xlabel, ylabel, title, legend, axis, save, path, save_data, outer_legend=True, normalize=normalize)

def plot_total_fluence_per_pass(amplifier, axis=None, save=False, save_path=None, save_data=False, show_title=True, custom_legend=""):
    """ 
    Plot the total fluence at the end of each pass
    """
    if amplifier.seed.CPA:
        amplifier.extraction_CPA()
    else:
        amplifier.extraction()

    fluence_out = amplifier.total_fluence_out
    x = np.arange(0,len(fluence_out),1)
    xlabel = "Pass number"
    ylabel = "output fluence in J/cm²"
    legend = custom_legend if custom_legend != "" else f"F$_{{max}}$ = {np.max(fluence_out)*1e-4:.3g}J/cm²"
    title  = f"{amplifier.crystal.material}, Output fluence vs. pass number" if show_title else None
    fname = f"Temporal_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}K_total-fluence.pdf"
    path = create_save_path(save_path, fname)

    kwargs = dict(marker="o")

    plot_function(x,fluence_out*1e-4, xlabel, ylabel, title, legend, axis, save, path, save_data, kwargs=kwargs)

def plot_inversion1D(amplifier, axis=None, save=False, save_path=None, save_data=False, ylim=(0,np.inf), show_title=True, custom_legend=""):
    """
    Plot the inversion in the crystal after the pumping process
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump
    x = crystal.z_axis*1e3
    y = crystal.inversion_end 
    xlabel = "depth $z$ in mm"
    ylabel = "inversion $\\beta$"
    legend = f"$\\beta_{{mean}}$ = {np.mean(crystal.inversion_end):.4f}" if custom_legend == "" else custom_legend
    title = "$\\beta(z)$ at the end of pumping" if show_title else None
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}kWcm2_inversion1D.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, ylim=ylim)

def plot_inversion_temporal(amplifier, axis=None, save=False, save_path=None, save_data=False, ylim=(0,np.inf), show_title=True, custom_legend=""):
    """
    Plot the inversion at the surface of the crystal as a function of time
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump
    x = pump.t_axis*1e3
    y = crystal.inversion[:,0]
    xlabel = "time $t$ in ms"
    ylabel = "inversion $\\beta$"
    legend = f"$\\beta_{{max}}$ = {np.ma.max(y):.4f}" if custom_legend == "" else custom_legend
    title = "$\\beta(t)$ at the surface of the medium" if show_title else None
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}kWcm2_inversion1D_front.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, ylim=ylim)

def plot_inversion2D(amplifier, cmap="magma", save=False, save_path=None, save_data=False, axis=None, show_title=True, custom_legend=""):
    """
    Plot the inversion in the crystal over time and space
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump

    if axis is None:
        fig, ax = plt.subplots()
    elif isinstance(axis, list):
        ax = axis[0]
        fig = axis[1]
    else:
        ax = axis

    extent = [0, crystal.z_axis[-1]*1e3, pump.duration*1e3, 0]
    im = ax.imshow(crystal.inversion, aspect='auto', extent=extent, cmap=cmap)
    # fig.colorbar(im, ax=ax, label="inversion $\\beta$")

    cbar = ax.figure.colorbar(im, ax=ax, use_gridspec=False, label="inversion $\\beta$")
    ax._colorbar = cbar  # store as attribute on the axis

    ax.set_ylabel("pump time $\\tau_p$ in ms")
    ax.set_xlabel("$z$ in mm")

    if custom_legend != "":
        base = ax.transAxes
        offset = offset_copy(base, x=10, y=10, units='points', fig=fig)
        ax.text(0, 0, custom_legend, transform=offset, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=1))
        
    if show_title:
        ax.set_title(r"$\beta$ vs time and space")
    ax.grid()
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_inversion2D.pdf"
    path = create_save_path(save_path, fname)

    if save:
        plt.tight_layout()
        plt.savefig(path)

def plot_spectral_fluence(amplifier, axis=None, save=False, save_path=None, save_data=False, xlim=(-np.inf,np.inf), show_title=True, normalize=False):
    """ 
    Plot the spectral fluence at the end of the ten last roundtrips. Note, that a roundtrip corresponds to two passes through the material.
    """
    spectral_fluence = amplifier.extraction_CPA()
    crystal, pump, seed = amplifier.crystal, amplifier.pump, amplifier.seed

    passes = len(spectral_fluence[:,0])-1

    x = seed.lambdas*1e9
    y_array = []
    legend = []
    for i in range(max(passes-20,0), passes+1, 2):
        y_array.append(spectral_fluence[i,:]*1e-9*1e-4)
        legend.append(f"$F =${integ(spectral_fluence[i,:], seed.dlambda)[-1]*1e-4:.3f}J/cm²")

    xlabel="wavelength $\\lambda$ in nm"
    ylabel="spectral fluence in J/cm²/nm"
    title = f"{crystal.name} with {int(passes/2)} RT, F$_0$ = {seed.fluence*1e-4:.2e} J/cm²" if show_title else None
    fname = f"Spectral_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}_spectral-fluence.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x,y_array, xlabel, ylabel, title, legend, axis, save, path, save_data, outer_legend=True, xlim=xlim, normalize=normalize)

def plot_inversion_before_after(amplifier, save=False, save_path=None, save_data=False, show_title=True, axis=None):
    """
    Plot the inversion before and after the amplification process
    """
    if amplifier.seed.CPA:
        amplifier.extraction_CPA()
    else:
        amplifier.extraction()

    x = amplifier.crystal.z_axis*1e3
    y_before = amplifier.crystal.inversion_end
    y_after = amplifier.beta_out
    xlabel = "depth $z$ in mm"
    ylabel = "inversion $\\beta$"
    legend = [f"$\\beta$ mean before = {np.mean(y_before):.4f}", f"$\\beta$ mean after = {np.mean(y_after):.4f}"]
    title = "$\\beta(z)$ before and after amplification" if show_title else None
    fname = f"{amplifier.crystal.material}_{amplifier.crystal.temperature}K_{amplifier.pump.intensity*1e-7}kWcm2_inversion_before_after.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, [y_before, y_after], xlabel, ylabel, title, legend, axis, save, path, save_data)

def plot_inversion_vs_pump_intensity(amplifier, axis=None, save=False, save_data=False, save_path=None, show_title=True, custom_legend=""):
    """
    Plot the inversion as a function of pump intensity
    """
    pump = amplifier.pump
    crystal = amplifier.crystal
    original_intensity = pump.intensity
    pump_intensities = np.linspace(0, 2, 20) * pump.intensity
    inversions = np.zeros_like(pump_intensities)

    for i, intensity in enumerate(pump_intensities):
        pump.intensity = intensity
        amplifier.inversion()
        inversions[i] = np.mean(crystal.inversion_end)

    x = pump_intensities*1e-7
    y = inversions
    xlabel = "pump intensity in kW/cm²"
    ylabel = "average inversion $\\beta$"
    title = "$\\beta(z)$ as a function of pump intensity" if show_title else None
    legend = None if custom_legend == "" else custom_legend
    fname = f"{crystal.material}_{crystal.temperature}K_{2*pump.intensity*1e-7}kWcm2_inversion_vs_pump_intensity.pdf"
    path = create_save_path(save_path, fname)
    kwargs = dict(marker="o")

    pump.intensity = original_intensity

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, kwargs=kwargs)

def plot_pump_absorption(amplifier, axis=None, save=False, save_data=False, save_path=None, show_title=True, custom_legend=""):
    """
    Plot the absorbed pump energy as a fraction of the total input fluence 
    """
    amplifier_ideal = copy.deepcopy(amplifier)
    amplifier_ideal.crystal.tau_f = np.inf # set an infinite lifetime to simulate ideal absorption
    amplifier_ideal.inversion()

    amplifier.inversion()

    crystal, pump = amplifier.crystal, amplifier.pump
    x = crystal.z_axis*1e3
    y = amplifier.pump.absorbed_energy
    y_ideal = amplifier_ideal.pump.absorbed_energy

    xlabel = "depth $z$ in mm"
    ylabel = "absorbed pump energy fraction $\\eta$"
    title = "absorbed pump light vs crystal thickness" if show_title else None
    legend = [f"$\\tau_f$ = {crystal.tau_f*1e3} ms, $\\eta_\\text{{max}}$ = {np.max(y)*1e2:.1f} %", f"$\\tau_f\\;\\to\\;\\infty$, $\\eta_\\text{{max}}$ = {np.max(y_ideal)*1e2:.1f} %"] if custom_legend == "" else custom_legend
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.duration*1e3}ms_{pump.intensity*1e-7}kWcm2_pump_absorption_vs_crystal_thickness.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, [y, y_ideal], xlabel, ylabel, title, legend, axis, save, path, save_data)

def plot_storage_efficiency_vs_pump_intensity(amplifier, axis=None, save=False, save_data=False, save_path=None, show_title=True, custom_legend=""):
    """
    Plot the storage efficiency as a function of pump intensity
    """
    pump = amplifier.pump
    crystal = amplifier.crystal
    pump_intensities = np.linspace(0, 2, 40) * pump.intensity

    storage_efficiency = amplifier.storage_efficiency(pump_intensity=pump_intensities)

    x = pump_intensities*1e-7
    y = storage_efficiency[:, -1]
    xlabel = "pump intensity in kW/cm²"
    ylabel = "storage efficiency"
    title = "storage efficiency as a function of pump intensity" if show_title else None
    legend = None if custom_legend == "" else custom_legend
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.duration*1e3}ms_storage_efficiency_vs_pump_intensity.pdf"
    path = create_save_path(save_path, fname)
    kwargs = dict(marker="o")

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, kwargs=kwargs)

def plot_storage_efficiency_vs_pump_time(amplifier, pump_intensity=[], axis=None, save=False, save_data=False, save_path=None, show_title=True, kwargs=None):
    """
    Plot the storage efficiency as a function of pump time
    pump_intensity: list of pump intensities in W/m²
    """
    time_factor = 2
    pump, crystal = amplifier.pump, amplifier.crystal
    pump_duration = time_factor*pump.duration
    if pump_intensity == []:
        pump_intensity.append(pump.intensity)

    storage_efficiency = amplifier.storage_efficiency(pump_intensity=pump_intensity, pump_duration=pump_duration)

    x = pump.t_axis*1e3
    y = [storage_efficiency[i,:] for i in range(len(storage_efficiency[:,0]))]
    xlabel = "pump duration in ms"
    ylabel = "storage efficiency"
    title = f"storage efficiency as a function of pump duration" if show_title else None
    legend = [f"$I_p$ = {intensity*1e-7:.1f} kW/cm²" for intensity in pump_intensity]
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}kWcm2_storage_efficiency_vs_pump_duration.pdf"
    path = create_save_path(save_path, fname)

    plot_function(x, y, xlabel, ylabel, title, legend, axis, save, path, save_data, kwargs=kwargs)

def plot_storage_efficiency_2D(amplifier, cmap="magma", save=False, save_path=None, save_data=False, show_title=True, custom_legend="", axis=None):
    """
    Plot the storage efficiency in the crystal over time and pump intensity
    """
    resolution_intensities = 41
    time_factor = 2

    pump, crystal = amplifier.pump, amplifier.crystal
    pump_intensities = np.linspace(0, 2, resolution_intensities) * pump.intensity
    pump_duration = time_factor*pump.duration

    storage_efficiency = amplifier.storage_efficiency(pump_intensity=pump_intensities, pump_duration=pump_duration)

    if axis is None:
        fig, ax = plt.subplots()
    elif isinstance(axis, list):
        ax = axis[0]
        fig = axis[1]
    else:
        ax = axis

    extent = [0, pump_duration*1e3, 0, pump_intensities[-1]*1e-7]
    # im = ax.imshow(storage_efficiency, aspect='auto', extent=extent, cmap=cmap, origin='lower')
    im = ax.contourf(storage_efficiency, extent=extent, cmap=cmap, origin='lower')

    cbar = ax.figure.colorbar(im, ax=ax, use_gridspec=False, label="storage efficiency", pad=0.01)
    ax._colorbar = cbar  # store as attribute on the axis

    ax.set_xlabel("pump duration $\\tau_p$ in ms")
    ax.set_ylabel("pump intensity $I_p$ in kW/cm²")
    if show_title:
        ax.set_title(r"storage efficiency vs pump duration and intensity") 

    if custom_legend != "":
        base = ax.transAxes
        offset = offset_copy(base, x=10, y=10, units='points', fig=fig)
        ax.text(0, 0, custom_legend, transform=offset, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ax.grid()
    fname = f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_storage_efficiency2D.pdf"
    path = create_save_path(save_path, fname)

    if save:
        plt.tight_layout()
        plt.savefig(path)

if __name__ == "__main__":
    crystal  = Crystal(material="YbCaF2", temperature=300, smooth_sigmas=True)

    CW_amplifier = Amplifier(crystal=crystal, seed=Seed())
    CPA_amplifier = Amplifier(crystal=crystal, seed=Seed_CPA())

    CW_amplifier.inversion()
    print(CW_amplifier)
    
    CPA_amplifier.inversion()
    plot_inversion1D(CW_amplifier)
    plot_inversion_temporal(CW_amplifier)
    plot_inversion2D(CW_amplifier)
    plot_total_fluence_per_pass(CW_amplifier)
    plot_temporal_fluence(CW_amplifier)
    plot_spectral_fluence(CPA_amplifier)
    plot_inversion_vs_pump_intensity(CW_amplifier)
    plot_inversion_before_after(CW_amplifier)
    plot_storage_efficiency_vs_pump_intensity(CW_amplifier)
    plot_storage_efficiency_2D(CW_amplifier)
    plot_storage_efficiency_vs_pump_time(CW_amplifier, pump_intensity=[10e7, 20e7, 30e7])
    plot_pump_absorption(CW_amplifier)

