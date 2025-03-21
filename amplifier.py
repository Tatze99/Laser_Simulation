from crystal import Crystal, plot_beta_eq
from pump import Pump
from seed import Seed
from seed_CPA import Seed_CPA
from spectral_losses import Spectral_Losses

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from utilities import z_integ, t_integ, integ, numres, h, c, set_plot_params
import os

set_plot_params()
Folder = os.path.dirname(os.path.abspath(__file__))

class Amplifier():

    def __init__(self, crystal=Crystal(), pump=Pump(), seed=Seed(), passes = 6, losses = 2e-2, print_iteration = False, spectral_losses = None, max_fluence=10):
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

        for k in range(self.passes):
		# calculate single components
            if k > 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            # equilibrium inversion and absorption coefficient at the seed wavelength
            beta_eq = self.crystal.beta_eq(self.seed.wavelength)
            alpha   = self.crystal.alpha(self.seed.wavelength)

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

        self.pulse_out = pulse_out
        self.temporal_fluence_out = pulse_out * c * h *c / self.seed.wavelength 
        self.fluence_out = z_integ(self.temporal_fluence_out, self.seed.dt)[:,-1]

        max_fluence = np.max(self.fluence_out)
        pump_fluence = self.pump.duration * self.pump.intensity
        max_gain = np.max(self.fluence_out[1::] / self.fluence_out[0:-1])

        print("Maximal laser fluence in J/cm²:", max_fluence/1e4)
        print("Total pump fluence in J/cm²:", pump_fluence/1e4)
        print("Maximal gain:", max_gain)
        
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

        Fsat = h*c /self.seed.lambdas /(self.crystal.sigma_a(self.seed.lambdas)+self.crystal.sigma_e(self.seed.lambdas))
        beta_eq = self.crystal.beta_eq(self.seed.lambdas)

        for k in range(self.passes):
		# calculate single components
            if k > 1:
                beta_out[k, :] = np.flipud(beta_out[k, :])

            for n, lambd in enumerate(self.seed.lambdas):

                Saturation = np.exp(spectral_fluence_out[k,n]*self.seed.dlambda/Fsat[n])
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

# =============================================================================
# Display of results
# =============================================================================

def plot_fluence(amplifier):
    """
    Plot the fluence at the end of each pass and the temporal fluence at the end of each pass.
    """
    fluence_out, pulse_out = amplifier.extraction()
    plt.figure()
    plt.plot(fluence_out*1e-4, "-o")
    plt.xlabel("Pass number")
    plt.ylabel("output fluence in J/cm²")
    plt.title('output fluence vs pass number')

    # plot the temporal fluence at the end of the last ten passes
    passes = len(pulse_out[:,0])-1
    plt.figure()
    for i in range(max(len(pulse_out[:,0])-10,0), len(pulse_out[:,0])):
        total_fluence = integ(pulse_out[i,:], amplifier.seed.dt)[-1]*1e-4
        plt.plot(amplifier.seed.time*1e9, pulse_out[i,:]*1e-9*1e-4, label=f"$F =$ {total_fluence:.3f} J/cm²")

    plt.xlabel("time $t$ in s")
    plt.ylabel("temporal fluence in J/cm²/ns")
    plt.title(f"{crystal.name} with {int(passes/2)} RT, F$_0$ = {seed.fluence*1e-4:.2e} J/cm²")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')


def plot_inversion1D(amplifier, save=False):
    """
    Plot the inversion in the crystal after the pumping process
    """
    amplifier.inversion()
    crystal, pump = amplifier.crystal, amplifier.pump

    plt.figure()
    plt.plot(crystal.z_axis*1e3, crystal.inversion_end, label=f"$\\beta$ mean = {np.mean(crystal.inversion_end):.4f}")
    plt.xlabel("depth $z$ in mm")
    plt.ylabel("inversion $\\beta$")
    plt.title('$\\beta(z)$ at the end of pumping')
    plt.ylim(bottom=0)
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_inversion1D.pdf"))

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

def plot_spectral_fluence(amplifier, lam_min = 1010, lam_max = 1050, save=False):
    """
    Plot the spectral fluence at the end of the ten last roundtrips. Note, that a roundtrip corresponds to two passes through the material.
    """
    spectral_fluence = amplifier.extraction_CPA()
    crystal, pump, seed = amplifier.crystal, amplifier.pump, amplifier.seed

    passes = len(spectral_fluence[:,0])-1
    plt.figure()
    for i in range(max(passes-20,0), passes+1, 2):
        total_fluence = integ(spectral_fluence[i,:], seed.dlambda)[-1]*1e-4
        plt.plot(seed.lambdas*1e9, spectral_fluence[i,:]*1e-9*1e-4, label=f"$F =$ {total_fluence:.3f} J/cm²")

    plt.xlabel("wavlength $\\lambda$ in nm")
    plt.ylabel("spectral fluence in J/cm²/nm")
    plt.xlim(lam_min, lam_max)
    plt.title(f"{crystal.name} with {int(passes/2)} RT, F$_0$ = {seed.fluence*1e-4:.2e} J/cm²")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}kW/cm2_{seed.fluence*1e-4}J/cm2_spectral_fluence.pdf"))
    
def test(amplifier):
    pump_intensites = np.linspace(0,50e7,20)
    efficiency = amplifier.storage_efficiency(pump_intensites)
    plt.figure()
    plt.plot(pump_intensites*1e-7, efficiency, 'o-')


if __name__ == "__main__":
    crystal  = Crystal(material="YbCaF2", temperature=300)
    plot_beta_eq(crystal,lambda_max=980e-9)
    crystal.smooth_cross_sections(0.9, 10, lambda_max=990e-9)
    pump     = Pump(intensity=39, wavelength=940, duration=crystal.tau_f*1e3)

    seed = Seed(fluence=0.01, duration=5, wavelength=1030, gauss_order=1, seed_type="gauss")
    CW_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed, passes=50, losses=1e-1)
    
    seed_CPA = Seed_CPA(fluence=2.7e-6, wavelength=1030, bandwidth=60, seed_type="rect")
    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=100, losses=1e-1, spectral_losses=None, max_fluence = 1)

    print(crystal,pump,seed_CPA,seed,sep='===========================\n')

    # test(CW_amplifier)
    
    CPA_amplifier.inversion()
    plot_inversion1D(CW_amplifier)
    plot_inversion2D(CW_amplifier)
    plot_fluence(CW_amplifier)
    plot_spectral_fluence(CPA_amplifier)

