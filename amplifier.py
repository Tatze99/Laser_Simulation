from crystal import Crystal
from pump import Pump
from seed import Seed

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as npm
from utilities import z_integ, t_integ, integ, numres, h, c


class Amplifier():

    def __init__(self):
        self.pump = Pump()
        self.seed = Seed()
        self.crystal = Crystal()
        self.passes = 6
        self.losses = 2e-2 

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
    plt.plot(amplifier.crystal.z_axis*1e3, amplifier.crystal.inversion_end)
    plt.xlabel("depth $z$ in mm")
    plt.ylabel("inversion $\\beta$")

def plot_inversion2D(amplifier):
    amplifier.inversion()

    plt.figure()
    ext = [0, amplifier.crystal.z_axis[-1], amplifier.seed.duration, 0]
    plt.imshow(amplifier.crystal.inversion, aspect='auto', extent=ext, cmap="magma")
    plt.colorbar(label="inversion")
    plt.ylabel("pump duration in s")
    plt.xlabel("z in m")
    plt.title(r'$\beta$ vs time and space')


if __name__ == "__main__":
    amplifier = Amplifier()
    amplifier.inversion()

    
    plot_inversion1D(amplifier)
    plot_inversion2D(amplifier)
    plot_fluence(amplifier)
