import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../"))
Folder = os.path.dirname(os.path.abspath(__file__))

from crystal import Crystal
from pump import Pump
from seed import Seed
from seed_CPA import Seed_CPA
from spectral_losses import Spectral_Losses
from amplifier import Amplifier
from utilities import integ

def gauss(x, a,b,c,d):
    return a*np.exp(-((x-b)/c)**2)+d

def plot_spectral_gain_losses(amplifier, losses, angles = None, mirror_losses = None, save = False):
    total_reflectivity = losses.reflectivity_by_angles(angles, angle_unit="deg")
    amplifier.spectral_losses = np.interp(amplifier.seed.lambdas, losses.lambdas, total_reflectivity)

    amplifier.spectral_losses += mirror_losses if mirror_losses is not None else 0
    spectral_fluence = amplifier.extraction_CPA()
    passes = len(spectral_fluence[:,0])-1
    gain_tot = np.max(spectral_fluence[-1,:])/np.max(spectral_fluence[0,:])
    plt.figure()
    plt.plot(amplifier.seed.lambdas*1e9, spectral_fluence[0,:]*1e-9*1e-4*gain_tot, '--', c="black", label=f"seed, G={gain_tot:.2e}")
    for i in range(max(passes-20,0), passes+1, 2):
        total_fluence = integ(spectral_fluence[i,:], amplifier.seed.dlambda)[-1]*1e-4
        plt.plot(amplifier.seed.lambdas*1e9, spectral_fluence[i,:]*1e-9*1e-4, label=f"$F =$ {total_fluence:.3f} J/cm²")

    plt.xlabel("wavlength in nm")
    plt.xlim(1010,1060)
    plt.ylabel("spectral fluence in J/cm²/nm")
    plt.title(f"{amplifier.crystal.name} with {passes} passes")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    if save and angles is not None:
        plt.savefig(os.path.join(Folder, f"TSF_Gain_{losses.TSF_name}_{amplifier.seed.fluence*1e-4:.1e}Jcm-2_{passes}pass_{angles[0]}_{angles[-1]}.pdf"))
    elif save:
        plt.savefig(os.path.join(Folder, f"TSF_Gain_{amplifier.crystal.name}_{amplifier.seed.fluence*1e-4:.1e}Jcm-2_{passes}pass_no_losses.pdf"))

    plt.figure()
    plt.plot(amplifier.seed.lambdas*1e9, amplifier.spectral_losses, label="total losses")
    if angles is not None:
        for angle in angles:
            plt.plot(amplifier.seed.lambdas*1e9, np.interp(amplifier.seed.lambdas, losses.lambdas, losses.calc_reflectivity(angle, angle_unit="deg")), label=f"loss $\\alpha = {angle}$°")
    if mirror_losses is not None:
        plt.plot(amplifier.seed.lambdas*1e9, mirror_losses, color="black", label="mirror losses")
    plt.xlim(1010,1060)
    plt.ylim(0,0.2)
    plt.legend()


def angle_variation_TSF(amplifier, losses, angle_low=43, angle_high=46, angle_step=1, number_of_TSF_pairs = 1, mirror_losses = 0, save = False):
    angles = np.arange(angle_low,angle_high+angle_step, angle_step)
    fig, axs = plt.subplots(len(angles), len(angles))
    fig.set_size_inches(16,16)
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0)
    for i, angle1 in enumerate(angles):
        for j, angle2 in enumerate(angles):
            angle_array = [angle1,angle2]*number_of_TSF_pairs
            total_reflectivity = 1.3*losses.reflectivity_by_angles(angle_array, angle_unit="deg")
            amplifier.spectral_losses = np.interp(amplifier.seed.lambdas, losses.lambdas, total_reflectivity)
            amplifier.spectral_losses += mirror_losses
            spectral_fluence = amplifier.extraction_CPA()
            roundtrips = int(0.5*(len(spectral_fluence[:,0])-1))
            axs[i,j].plot(amplifier.seed.lambdas*1e9, spectral_fluence[-1,:]/np.max(spectral_fluence[-1,:]), label=f"{angle1}°, {angle2}°, {roundtrips} RT")
            
            if i == len(angles)-1:
                axs[i,j].set_xlabel("$\\lambda$ in nm")
            else:
                axs[i,j].set_xticklabels([])
            if j > 0:
                axs[i,j].set_yticklabels([])

            axs[i,j].set_xlim(1010.1,1059.9)
            axs[i,j].set_ylim(-0.1,1.25)
            axs[i,j].grid()
            axs[i,j].legend(loc="upper right")

    if save:
        plt.savefig(os.path.join(Folder, f"TSF_Angle_variation_{losses.TSF_name}_{angle_low}_{angle_high}_{len(angle_array)}TSF.pdf"))

def plot_small_signal_gain(amplifier, beta, measured_gain_name, losses = 0, mirror_losses = 0, lam_min=1010, lam_max=1060):
    measured_gain = np.loadtxt(os.path.join(Folder, measured_gain_name))
    plt.figure()
    lambd = amplifier.seed.lambdas

    # make multiple plots if beta is an array!
    if isinstance(beta, (list, tuple, np.ndarray)): 
        for b in beta:
            Gain = amplifier.crystal.small_signal_gain(lambd, b)**2*(1-losses)
            
            Gain *= (1-mirror_losses)
            plt.plot(lambd*1e9, Gain, label=f"$\\beta$ = {b:.2f}")
    # if beta is just a number, plot a single plot
    else:
        Gain = amplifier.crystal.small_signal_gain(lambd, beta)
        plt.plot(lambd*1e9, Gain, label=f"$\\beta$ = {beta:.2f}")
    
    plt.plot(measured_gain[:,0], measured_gain[:,1], label="measured gain", color="black")
    plt.xlabel("wavelength in nm")
    plt.ylabel("Gain G")
    
    plt.xlim(lam_min,lam_max)
    plt.ylim(bottom=1.1)
    plt.title(f"small signal gain, {amplifier.crystal.name} at {amplifier.crystal.temperature}K")
    plt.legend()

def simulate_YbFP15():
    crystal  = Crystal(material="YbFP15_Toepfer")
    crystal.smooth_cross_sections(0, 1, lambda_min=1050e-9)
    pump     = Pump(intensity=23, wavelength=940, duration=2)
    seed_CPA = Seed_CPA(fluence=2.7e-9, wavelength=1030, bandwidth=70, seed_type="rect")
    spectral_losses   = Spectral_Losses(material="YbFP15")

    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=250, losses=1.3e-1, spectral_losses=None, max_fluence = 1)

    CPA_amplifier.inversion()
    plot_small_signal_gain(CPA_amplifier, [0.174], "240626_FP15_Q-Switch_TFP_Gain100A.txt", losses=1.3e-1)
    plot_spectral_gain_losses(CPA_amplifier, spectral_losses, [43.2,43.2,46,46])
    # angle_variation_TSF(CPA_amplifier, spectral_losses, angle_low=43, angle_high=47, angle_step=1, save=True)

def simulate_CaF2():
    crystal  = Crystal(material="YbCaF2_Toepfer")
    crystal.smooth_cross_sections(0.9, 10)
    pump     = Pump(intensity=39, wavelength=920, duration=4)
    seed_CPA = Seed_CPA(fluence=2.7e-9, wavelength=1035, bandwidth=70, seed_type="rect")
    spectral_losses   = Spectral_Losses(material="YbCaF2_Garbsen")

    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=200, losses=0.7e-1, spectral_losses=None, max_fluence = 1)
    CPA_amplifier.inversion()

    mirror_losses = gauss(CPA_amplifier.seed.lambdas, 0.1, 1010e-9, 10e-9, 0)

    # plot_small_signal_gain(CPA_amplifier, [0.3], "240719_Q-Switch_CaF2_TFP_Gain100A.txt", losses=0.7e-1, mirror_losses=mirror_losses)
    plot_spectral_gain_losses(CPA_amplifier, spectral_losses, angles = [45,45, 45, 45], mirror_losses = mirror_losses, save=True)
    # angle_variation_TSF(CPA_amplifier, spectral_losses, angle_low=44, angle_high=46, angle_step=0.5, save=True, mirror_losses=mirror_losses, number_of_TSF_pairs=1)


def simulate_A2_CaF2():
    # Rechnungen auf Seite 52 im Notebook
    crystal  = Crystal(material="YbCaF2_Toepfer")
    crystal.smooth_cross_sections(0.9, 10)
    pump     = Pump(intensity=39, wavelength=920, duration=4)
    seed_CPA = Seed_CPA(fluence=3.2e-4, wavelength=1035, bandwidth=20, seed_type="gauss")
    spectral_losses   = Spectral_Losses(material="YbCaF2_Garbsen")

    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=200, losses=0.7e-1, spectral_losses=None, max_fluence = 1)
    CPA_amplifier.inversion()

    mirror_losses = gauss(CPA_amplifier.seed.lambdas, 0.1, 1010e-9, 10e-9, 0)

    CPA_amplifier.plot_inversion1D()
    # plot_small_signal_gain(CPA_amplifier, [0.3], "240719_Q-Switch_CaF2_TFP_Gain100A.txt", losses=0.7e-1, mirror_losses=mirror_losses)
    # plot_spectral_gain_losses(CPA_amplifier, spectral_losses, angles = [45,45, 45, 45], mirror_losses = mirror_losses, save=True)
    # angle_variation_TSF(CPA_amplifier, spectral_losses, angle_low=44, angle_high=46, angle_step=0.5, save=True, mirror_losses=mirror_losses, number_of_TSF_pairs=1)


if __name__ == "__main__":

    simulate_A2_CaF2()
    # simulate_CaF2()
    # simulate_YbFP15()

    
    