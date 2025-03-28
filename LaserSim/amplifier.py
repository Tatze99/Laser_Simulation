from LaserSim.crystal import Crystal, plot_beta_eq
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.spectral_losses import Spectral_Losses
from LaserSim.utilities import z_integ, t_integ, integ, numres, h, c, set_plot_params, plot_function

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
import os

set_plot_params()
Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

class Amplifier():

    def __init__(self, crystal=Crystal(), pump=Pump(), seed=Seed(), passes = 50, losses = 2e-2, print_iteration = False, spectral_losses = None, max_fluence=10):
        self.pump = pump
        self.seed = seed
        self.crystal = crystal
        self.passes = passes
        self.losses = losses
        self.print_iteration = print_iteration
        self.spectral_losses = spectral_losses
        self.max_fluence = max_fluence*1e4  # [J/m²]

    def inversion(self, pump_intensity = None):

        if pump_intensity is None:
            pump_intensity = self.pump.intensity
        beta_eq = self.crystal.beta_eq(self.pump.wavelength)
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
            max_it = 20
            abweichung = np.abs(np.max(beta_high-beta_low))
            fehler = 1/numres
            A = (it > max_it)
            B = (abweichung < fehler)
            if A and self.print_iteration:
                print("Max iterations of " + str(it) + " exceeded.")
            if B and self.print_iteration:
                print("Deviation between beta_low and beta_high =",
                    abweichung, "<", fehler)
            return A or B

        # Initialisierung
        beta_low = np.zeros((numres, numres))
        beta_high = np.ones((numres, numres)) * beta_eq
        R0 = pump_intensity * sigma_a / h / (c / self.pump.wavelength)
        lambert_beer = np.exp(-alpha * z_axis)
        Ra = R0 * npm.repmat(lambert_beer, numres, 1)
        iteration = 0
        betas = []
        pumprates = []

        # Iteration
        while not break_condition(beta_low, beta_high, iteration):
            iteration += 1
            if self.print_iteration: print("Iteration Nummer: " + str(iteration))
            R_high = Ra * Rb(beta_high)
            R_low = Ra * Rb(beta_low)
            beta_high = beta(R_high)
            beta_low = beta(R_low)

            betas.append((beta_low, beta_high))
            pumprates.append((R_low, R_high))
    
        beta_end = (beta_low[-1,:] + beta_high[-1,:]) / 2
        beta_total = (beta_low + beta_low) / 2
        pumprate = (R_high + R_low) / 2

        stored_fluence = h * c / self.seed.wavelength * self.crystal.doping_concentration * np.sum(beta_total - self.crystal.beta_eq(self.seed.wavelength), 1) * self.crystal.dz

        self.crystal.inversion = beta_total
        self.pump.pumprate = pumprate
        self.crystal.inversion_end = beta_end
        self.crystal.stored_fluence = stored_fluence

        return beta_end

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
        break_index = self.passes

        # equilibrium inversion and absorption coefficient at the seed wavelength
        beta_eq = self.crystal.beta_eq(self.seed.wavelength)
        alpha   = self.crystal.alpha(self.seed.wavelength)
        
        for k in range(self.passes):
		# calculate single components
            if k > 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            # Gain G(z), spatial array of the integrated small signal gain
            Gain = np.exp(-alpha * integ(1-beta_out[k,:]/beta_eq, self.crystal.dz))

            # Saturation S(t), temporal array of the saturation, Saturation_end: S(t=t_end)
            Saturation = np.exp(sigma_tot * c * integ(pulse_out[k, :], self.seed.dt))
    
            # compute temporal pulse shape after amplification at z = length
            pulse_out[k+1,:] = pulse_out[k,:] * (Saturation*Gain[-1])/(1+(Saturation-1)*Gain[-1]) * (1-self.losses)

            # compute beta after amplification at t = t_end
            beta_out[k+1,:] = beta_eq + (beta_out[k,:] - beta_eq) / (1 + (Saturation[-1] - 1)*Gain)

            if k % 2:
                beta_out[k+1,:] = np.flipud(beta_out[k+1,:])

            if integ(pulse_out[k+1,:], self.seed.dt)[-1]* c * h *c / self.seed.wavelength > self.max_fluence:
                break_index = k+1
                break

        self.pulse_out = pulse_out[:break_index+1,:]
        self.temporal_fluence_out = self.pulse_out * c * h *c / self.seed.wavelength 
        self.fluence_out = z_integ(self.temporal_fluence_out, self.seed.dt)[:,-1]
        self.max_gain = np.max(self.fluence_out[1::] / self.fluence_out[0:-1])

        return self.fluence_out, self.temporal_fluence_out

    # =============================================================================
    # Calculate energy extraction for a CPA pulse
    # =============================================================================

    def extraction_CPA(self):
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

        for k in range(self.passes):
		# calculate single components
            if k > 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            # Compute the wavelength dependent saturation
            Saturation = np.exp(integ(spectral_fluence_out[k,:], self.seed.dlambda)[-1]/Fsat)

            # Compute the small signal gain by integrating along z
            Gain = np.exp(self.crystal.doping_concentration*z_integ(np.outer(sigma_e,beta_out[k,:])-np.outer(sigma_a,(1-beta_out[k,:])),self.crystal.dz))[:,-1]

            # Calculate the spectral fluence using Frantz Nodvik
            spectral_fluence_out[k+1, :] = spectral_fluence_out[k, :] * Fsat/(integ(spectral_fluence_out[k,:], self.seed.dlambda)[-1]) * np.log(1+Gain*(Saturation-1))

            if k % 2:
                spectral_fluence_out[k+1, :] *= (1-self.losses-self.spectral_losses)
            beta_out[k+1,:] = np.mean(beta_eq) + (beta_out[k,:]-np.mean(beta_eq))/(1+np.mean(Gain)*(np.mean(Saturation)-1))
            
            if integ(spectral_fluence_out[k+1,:], self.seed.dlambda)[-1] > self.max_fluence:
                return spectral_fluence_out[~np.all(spectral_fluence_out == 0, axis=1)]
        return spectral_fluence_out

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

        Fsat = h*c /self.seed.lambdas /(self.crystal.sigma_a(self.seed.lambdas)+self.crystal.sigma_e(self.seed.lambdas))
        beta_eq = self.crystal.beta_eq(self.seed.lambdas)

        for k in range(self.passes):
		# calculate single components
            if k > 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            for n, lambd in enumerate(self.seed.lambdas):
                # Saturation = np.exp(spectral_fluence_out[k,n]*self.seed.dlambda/Fsat[n])
                Saturation = np.exp((self.crystal.sigma_a(lambd)+self.crystal.sigma_e(lambd)) * c * (spectral_fluence_out[k,n] /(h*c**2)*lambd) * self.seed.dlambda)
                if n == int(len(self.seed.lambdas)/2): print(spectral_fluence_out[k,n]*len(self.seed.lambdas)*self.seed.dlambda*1e-4, Fsat[n]*1e-4, Saturation, lambd)
                Gain = np.exp(self.crystal.doping_concentration*integ(self.crystal.sigma_e(lambd)*beta_out[k,:]-(1-beta_out[k,:])*self.crystal.sigma_a(lambd),self.crystal.dz))

                spectral_fluence_out[k+1, n] = Fsat[n] * np.log(1+Gain[-1]*(Saturation-1))/self.seed.dlambda

                if k % 2:
                    spectral_fluence_out[k+1, n] *= (1-self.losses-self.spectral_losses[n])
                beta_out[k+1,:] = beta_eq[n] + (beta_out[k,:]-beta_eq[n])/(1+Gain*(Saturation-1))
            
            if integ(spectral_fluence_out[k+1,:], self.seed.dlambda)[-1] > self.max_fluence:
                return spectral_fluence_out[~np.all(spectral_fluence_out == 0, axis=1)]
        return spectral_fluence_out

    def storage_efficiency(self, pump_intensities):
        """
        Calculate the storage efficiency of the crystal.
        """

        inversion_array = np.zeros((len(pump_intensities),len(self.crystal.z_axis)))

        for i, pump_intensity in enumerate(pump_intensities):
            inversion_array[i,:] = self.inversion(pump_intensity=pump_intensity)

        extractable_fluence = z_integ(inversion_array - self.crystal.beta_eq(self.seed.wavelength), self.crystal.dz)[:,-1] * h * c * self.crystal.doping_concentration / (self.seed.wavelength)

        efficiency = extractable_fluence / (pump_intensities * self.pump.duration)
        efficiency = np.where(efficiency < 0, 0, efficiency)

        return efficiency.T
    
    def __repr__(self):
        return (f"{self.crystal}{self.pump}{self.seed}"
                f"Amplifier:\n"
                f"- medium passes = {self.passes}\n"
                f"- losses = {self.losses*1e2}% \n"
                f"- maximum allowed fluence = {self.max_fluence*1e-4}J/cm²\n"
        )
# =============================================================================
# Display of results
# =============================================================================

def plot_temporal_fluence(amplifier, save=False, save_path=None):
    """ 
    Plot the temporal fluence for the last ten passes
    """
    _, pulse_out = amplifier.extraction()
    passes = len(pulse_out[:,0])-1

    x = amplifier.seed.time*1e9
    y_array = []
    legend = []
    for i in range(max(len(pulse_out[:,0])-10,0), len(pulse_out[:,0])):
        y_array.append(pulse_out[i,:]*1e-9*1e-4)
        legend.append(f"$F =${integ(pulse_out[i,:], amplifier.seed.dt)[-1]*1e-4:.3f}J/cm²")

    xlabel="time $t$ in s"
    ylabel="temporal fluence in J/cm²/ns"
    title = f"{amplifier.crystal.name} with {int(passes/2)} RT, F$_0$ = {amplifier.seed.fluence*1e-4:.2e} J/cm²"
    path = save_path or os.path.join(Folder, "material_database", "plots", f"Temporal_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}_temporal-fluence.pdf")
    kwargs = dict(marker="o")

    plot_function(x,y_array, xlabel, ylabel, title, legend, save, path, outer_legend=True)


def plot_total_fluence_per_pass(amplifier, save=False, save_path=None):
    """ 
    Plot the total fluence at the end of each pass
    """
    fluence_out, _ = amplifier.extraction()
    x = np.arange(0,len(fluence_out),1)
    xlabel = "Pass number"
    ylabel = "output fluence in J/cm²"
    legend = f"F$_{{max}}$ = {np.max(fluence_out)*1e-4:.3g}J/cm²"
    title  = f"{amplifier.crystal.material}, Output fluence vs. pass number"
    path = save_path or os.path.join(Folder, "material_database", "plots", f"Temporal_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}K_total-fluence.pdf")
    kwargs = dict(marker="o")

    plot_function(x,fluence_out*1e-4, xlabel, ylabel, title, legend, save, path, kwargs=kwargs)


def plot_inversion1D(amplifier, save=False, save_path=None, ylim=(0,np.inf)):
    """
    Plot the inversion in the crystal after the pumping process
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump
    x = crystal.z_axis*1e3
    y = crystal.inversion_end 
    xlabel = "depth $z$ in mm"
    ylabel = "inversion $\\beta$"
    legend = f"$\\beta$ mean = {np.mean(crystal.inversion_end):.4f}"
    title = "$\\beta(z)$ at the end of pumping"
    path = save_path or os.path.join(Folder, "material_database", "plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}kWcm2_inversion1D.pdf")

    plot_function(x, y, xlabel, ylabel, title, legend, save, path, ylim=ylim)


def plot_inversion2D(amplifier, cmap="magma", save=False):
    """
    Plot the inversion in the crystal over time and space
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump

    plt.figure()
    extent = [0, crystal.z_axis[-1]*1e3, pump.duration*1e3, 0]
    plt.imshow(crystal.inversion, aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(label="inversion $\\beta$")
    plt.ylabel("pump time $\\tau_p$ in ms")
    plt.xlabel("$z$ in mm")
    plt.title(r"$\beta$ vs time and space")
    plt.grid()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_inversion2D.pdf"))


def plot_spectral_fluence(amplifier, save=False, save_path=None, xlim=(1010,1050)):
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
    title = f"{crystal.name} with {int(passes/2)} RT, F$_0$ = {seed.fluence*1e-4:.2e} J/cm²"
    path = save_path or os.path.join(Folder, "material_database", "plots", f"Spectral_seed_{amplifier.seed.fluence*1e-4}Jcm2_{amplifier.crystal.material}_{amplifier.crystal.temperature}_spectral-fluence.pdf")

    plot_function(x,y_array, xlabel, ylabel, title, legend, save, path, outer_legend=True, xlim=xlim)


if __name__ == "__main__":
    crystal  = Crystal(material="YbCaF2", temperature=300, smooth_sigmas=True)

    CW_amplifier = Amplifier(crystal=crystal,)
    CPA_amplifier = Amplifier(crystal=crystal, seed=Seed_CPA())

    print(CW_amplifier)
    
    CPA_amplifier.inversion()
    plot_inversion1D(CW_amplifier)
    plot_inversion2D(CW_amplifier)
    plot_total_fluence_per_pass(CW_amplifier)
    plot_temporal_fluence(CW_amplifier)
    plot_spectral_fluence(CPA_amplifier)

