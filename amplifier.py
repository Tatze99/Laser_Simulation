from crystal import Crystal
from pump import Pump
from seed import Seed
from seed_CPA import Seed_CPA

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from utilities import z_integ, t_integ, integ, numres, h, c


class Amplifier():

    def __init__(self, material="YbCaF2", pump=Pump(), seed=Seed(), passes = 6, losses = 2e-2):
        self.pump = pump
        self.seed = seed
        self.crystal = Crystal(material=material)
        self.passes = passes
        self.losses = losses

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
            if A:
                print("Max iterations of " + str(it) + " exceeded.")
            if B:
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
            print("Iteration Nummer: " + str(iteration))
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

    ################################################################
    ################################################################
    ################################################################
    
    def extraction(self):
        if np.any(self.crystal.inversion_end):
            beta_0 = self.crystal.inversion_end           
        else:
            beta_0 = self.inversion()

        stepsize = self.seed.stepsize

        pulse_out = np.zeros((self.passes+1, len(self.seed.pulse)))
        pulse_out[0,:] = self.seed.pulse
        beta_out = np.zeros( (self.passes+1, len(beta_0)) )
        beta_out[0,:] = np.abs(beta_0)
        fluence_out = np.zeros(self.passes+1)
        fluence_out[0] =  np.sum(pulse_out[0,:] ) * stepsize * c * h *c / self.seed.wavelength
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
            Saturation = np.exp(sigma_tot * c * integ(pulse_out[k, :], self.seed.stepsize))

            # compute temporal pulse shape after amplification at z = length
            pulse_out[k+1,:] = pulse_out[k,:] * (Saturation*Gain[-1])/(1+(Saturation-1)*Gain[-1]) * (1-self.losses)

            # compute fluence after amplification
            fluence_out[k+1] =  np.sum(pulse_out[k+1,:] ) * stepsize * c * h * c / self.seed.wavelength # J/m²

            # compute beta after amplification at t = t_end
            beta_out[k+1,:] = beta_eq + (beta_out[k,:] - beta_eq) / (1 + (Saturation[-1] - 1)*Gain) * (1-self.losses)

            if k % 2:
                beta_out[k+1,:] = np.flipud(beta_out[k+1,:])

        self.fluence_out = fluence_out
        self.pulse_out = pulse_out

        max_fluence = np.max(fluence_out)
        pump_fluence = self.pump.duration * self.pump.intensity
        max_gain = np.max(fluence_out[1::] / fluence_out[0:-1])
        print()
        print("Maximal laser fluence in J/cm^2:", max_fluence/1e4)
        print("Total pump fluence in J/cm^2:", pump_fluence/1e4)
        print("Maximal gain:", max_gain)
        print()
        
        return fluence_out, pulse_out

    def extraction_CPA(self):
        if np.any(self.crystal.inversion_end):
            beta_0 = self.crystal.inversion_end           
        else:
            beta_0 = self.inversion()
        
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
                # total_beta = integ(beta_out[k,:], self.crystal.dz)
                Saturation = np.exp(spectral_fluence_out[k,n]/Fsat[n])
                Gain = np.exp(self.crystal.doping_concentration*integ(self.crystal.sigma_e(lambd)*beta_out[k,:]-(1-beta_out[k,:])*self.crystal.sigma_a(lambd),self.crystal.dz))

                spectral_fluence_out[k+1, n] = Fsat[n] * np.log(1+Gain[-1]*(Saturation-1))
                beta_out[k+1,:] = beta_eq[n] + (beta_out[k,:]-beta_eq[n])/(1+Gain*(Saturation-1))
        
                # print(n, k, Gain[-1], np.exp(self.seed.spectral_fluence[n]/Fsat[n]))
        return spectral_fluence_out
# =============================================================================
# Display of results
# =============================================================================

def plot_fluence(amplifier):
    fluence_out, pulse_out = amplifier.extraction()

    plt.figure()
    plt.plot(fluence_out*1e-4, "-o")
    plt.xlabel("Pass number")
    plt.ylabel("Fluence in J/cm^2")
    plt.grid()

def plot_inversion1D(amplifier):
    amplifier.inversion()

    plt.figure()
    plt.plot(amplifier.crystal.z_axis*1e3, amplifier.crystal.inversion_end, label=f"$\\beta$ mean = {np.mean(amplifier.crystal.inversion_end):.4f}")
    plt.xlabel("depth $z$ in mm")
    plt.ylabel("inversion $\\beta$")
    plt.title('$\\beta(z)$ at the end of pumping')
    plt.legend()
    print(amplifier.crystal.inversion_end[0])

def plot_inversion2D(amplifier):
    amplifier.inversion()

    plt.figure()
    ext = [0, amplifier.crystal.z_axis[-1], amplifier.seed.duration, 0]
    plt.imshow(amplifier.crystal.inversion, aspect='auto', extent=ext, cmap="magma")
    plt.colorbar(label="inversion")
    plt.ylabel("pump duration in s")
    plt.xlabel("z in m")
    plt.title(r'$\beta$ vs time and space')

def plot_spectral_gain(amplifier):
    spectral_fluence = amplifier.extraction_CPA()

    plt.figure()
    for i in range(max(amplifier.passes-10,0), amplifier.passes):
        # plt.plot(amplifier.seed.lambdas*1e9, spectral_fluence[i,:]*1e-4, label=i)
        plt.plot(amplifier.seed.lambdas*1e9, spectral_fluence[i,:]/np.max(spectral_fluence[0,:]), label=i)
    Gain = amplifier.crystal.small_signal_gain(amplifier.seed.lambdas, np.mean(amplifier.crystal.inversion_end))
    plt.plot(amplifier.seed.lambdas*1e9, Gain**(amplifier.passes-1), '--',label=f"small signal gain")
    plt.xlabel("wavlength in nm")
    plt.ylabel("fluence in J/cm²")
    plt.legend()
    plt.grid()

if __name__ == "__main__":
    pump = Pump(intensity =30, wavelength=940, duration=2)
    seed = Seed_CPA(fluence=1)
    amplifier = Amplifier(material="YbFP15_Toepfer", pump=pump, seed=seed, passes=2, losses=2e-2)
    amplifier.inversion() 

    plot_spectral_gain(amplifier)
    plot_inversion1D(amplifier)
    # plot_inversion2D(amplifier)
    # plot_fluence(amplifier)
