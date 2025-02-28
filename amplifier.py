from crystal import Crystal
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

    def inversion(self):
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
        R0 = self.pump.intensity * sigma_a / h / (c / self.pump.wavelength)
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
        storage_efficiency = np.array(stored_fluence)
        # storage_efficiency[0] = -1
        # storage_efficiency[1::] = np.clip(stored_fluence[1::] / t_axis[1::] / self.pump.intensity, -1, 10.0)
        self.crystal.inversion = beta_total
        self.pump.pumprate = pumprate
        self.crystal.inversion_end = beta_end
        self.crystal.stored_fluence = stored_fluence
        self.crystal.storage_efficiency = storage_efficiency

        return beta_end

    # =============================================================================
    # Calculate energy extraction for a monochromatic, temporal pulse
    # =============================================================================
    
    def extraction(self):
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

# =============================================================================
# Display of results
# =============================================================================

def plot_fluence(amplifier):
    fluence_out, pulse_out = amplifier.extraction()

    plt.figure()
    plt.plot(fluence_out*1e-4, "-o")
    plt.xlabel("Pass number")
    plt.ylabel("output fluence in J/cm²")
    plt.title('output fluence vs pass number')

    plt.figure()
    for i in range(max(len(pulse_out[0,:])-10,0), len(pulse_out[0,:])+1):
        total_fluence = integ(pulse_out[i,:], amplifier.seed.dt)[-1]*1e-4
        plt.plot(amplifier.seed.time*1e9, pulse_out[i,:]*1e-9*1e-4, label=f"$F =$ {total_fluence:.3f} J/cm²")

    plt.xlabel("time $t$ in s")
    plt.ylabel("temporal fluence in J/cm²/ns")
    plt.legend()

def plot_inversion1D(amplifier, save=False):
    amplifier.inversion()
    crystal = amplifier.crystal 
    pump = amplifier.pump

    plt.figure()
    plt.plot(crystal.z_axis*1e3, crystal.inversion_end, label=f"$\\beta$ mean = {np.mean(crystal.inversion_end):.4f}")
    plt.xlabel("depth $z$ in mm")
    plt.ylabel("inversion $\\beta$")
    plt.title('$\\beta(z)$ at the end of pumping')
    plt.legend()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_inversion1D.pdf"))

def plot_inversion2D(amplifier, save=False):
    amplifier.inversion()

    plt.figure()
    ext = [0, amplifier.crystal.z_axis[-1]*1e3, amplifier.pump.duration*1e3, 0]
    plt.imshow(amplifier.crystal.inversion, aspect='auto', extent=ext, cmap="magma")
    plt.colorbar(label="inversion $\\beta$")
    plt.ylabel("pump time in ms")
    plt.xlabel("z in mm")
    plt.title(r'$\beta$ vs time and space')
    plt.grid()

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{crystal.material}_{crystal.temperature}K_{pump.intensity*1e-7}_inversion1D.pdf"))

def plot_spectral_gain(amplifier):
    spectral_fluence = amplifier.extraction_CPA()
    passes = len(spectral_fluence[:,0])-1
    plt.figure()
    for i in range(max(passes-20,0), passes+1, 2):
        total_fluence = integ(spectral_fluence[i,:], amplifier.seed.dlambda)[-1]*1e-4
        plt.plot(amplifier.seed.lambdas*1e9, spectral_fluence[i,:]*1e-9*1e-4, label=f"$F =$ {total_fluence:.3f} J/cm²")

    plt.xlabel("wavlength in nm")
    plt.xlim(1010,1050)
    plt.ylabel("spectral fluence in J/cm²/nm")
    plt.title(f"{crystal.name} with {passes} passes")
    plt.legend()
    
def simulate_YbFP15():
    crystal  = Crystal(material="YbFP15_Toepfer")
    pump     = Pump(intensity=23, wavelength=940, duration=2)
    seed_CPA = Seed_CPA(fluence=1e-6, wavelength=1030, bandwidth=60, seed_type="gauss")
    losses   = Spectral_Losses(material="YbFP15")

    angle1 = 47
    angle2 = 43.3
    total_reflectivity = losses.reflectivity_by_angles([angle1,angle2,angle1,angle2], angle_unit="deg")
    spectral_losses = np.interp(seed_CPA.lambdas, losses.lambdas, total_reflectivity)

    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=200, losses=1.3e-1, spectral_losses=None, max_fluence = 1)

    CPA_amplifier.inversion()

    plot_spectral_gain(CPA_amplifier)

    plt.figure()
    plt.plot(seed_CPA.lambdas*1e9, spectral_losses)
    plt.xlim(1010,1050)


if __name__ == "__main__":

    crystal  = Crystal(material="YbCaF2_Toepfer")
    crystal.smooth_cross_sections(0.9, 10)
    pump     = Pump(intensity=39, wavelength=920, duration=4)
    seed_CPA = Seed_CPA(fluence=2.7e-6, wavelength=1030, bandwidth=60, seed_type="rect")
    losses   = Spectral_Losses(material="YbCaF2_Garbsen")
    # print(crystal,pump,seed_CPA,seed,sep='')


    CW_amplifier  = Amplifier(crystal=crystal, pump=pump, passes=60, losses=1e-1)
    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=100, losses=1e-1, spectral_losses=None, max_fluence = 1)

    CPA_amplifier.inversion()
    plot_inversion1D(CW_amplifier)
    # plot_inversion2D(CW_amplifier)
    # plot_fluence(CW_amplifier)

    # plot_spectral_gain(CPA_amplifier)
