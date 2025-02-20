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


def plot_spectral_gain_losses(amplifier, angles = [45, 45, 45, 45]):
    total_reflectivity = losses.reflectivity_by_angles(angles, angle_unit="deg")
    amplifier.spectral_losses = np.interp(seed_CPA.lambdas, losses.lambdas, total_reflectivity)

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

    plt.figure()
    plt.plot(seed_CPA.lambdas*1e9, amplifier.spectral_losses)
    for angle in angles:
        plt.plot(seed_CPA.lambdas*1e9, np.interp(seed_CPA.lambdas, losses.lambdas, losses.calc_reflectivity(angle, angle_unit="deg")))
    plt.xlim(1010,1050)
    plt.ylim(0,0.2)

def angle_variation_TSF(amplifier, save = False):
    angle_low = 43
    angle_high = 47
    angles = np.arange(angle_low,angle_high+1,1)
    fig, axs = plt.subplots(len(angles), len(angles))
    fig.set_size_inches(16,16)
    fig.subplots_adjust(left=0.03, bottom=0.03, right=0.99, top=0.99, wspace=0, hspace=0)
    for i, angle1 in enumerate(angles):
        for j, angle2 in enumerate(angles):
            total_reflectivity = losses.reflectivity_by_angles([angle1, angle2], angle_unit="deg")
            amplifier.spectral_losses = np.interp(seed_CPA.lambdas, losses.lambdas, total_reflectivity)
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
        plt.savefig(os.path.join(Folder, f"TSF_Angle_variation_{angle_low}_{angle_high}.pdf"))


if __name__ == "__main__":
    crystal  = Crystal(material="YbCaF2_Toepfer")
    crystal.smooth_cross_sections(0.9, 10)
    pump     = Pump(intensity=32, wavelength=920, duration=4)
    seed_CPA = Seed_CPA(fluence=2.7e-6, wavelength=1035, bandwidth=70, seed_type="rect")
    losses   = Spectral_Losses(material="YbCaF2_Garbsen")
    # print(crystal,pump,seed_CPA,seed,sep='')

    
    CPA_amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, passes=150, losses=1e-1, spectral_losses=None, max_fluence = 1)
    CPA_amplifier.inversion()

    # angle = 45
    # plot_spectral_gain_losses(CPA_amplifier, angles=[angle,angle,angle,angle])

    angle_variation_TSF(CPA_amplifier, save=True)

    # CPA_amplifier.inversion()
    # plot_inversion1D(CW_amplifier)
    # plot_inversion2D(CW_amplifier)
    # plot_fluence(CW_amplifier)


    