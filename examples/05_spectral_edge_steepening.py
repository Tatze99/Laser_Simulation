import numpy as np
from matplotlib import pyplot as plt
from LaserSim.crystal import Crystal, plot_small_signal_gain
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.amplifier import Amplifier, plot_total_fluence_per_pass, plot_temporal_fluence, plot_spectral_fluence, plot_inversion1D


### YbCAF2 ###
crystal = Crystal(material="YbCaF2", smooth_sigmas=True)
pump = Pump(intensity=40, duration=4)
passes = 100
losses = 0.078

### YbFP15 ###
crystal = Crystal(material="YbFP15_Toepfer", smooth_sigmas=True)
pump = Pump(intensity=20, duration=2.7)
passes = 55
losses = 0.12


max_fluence = 10    # in J/cm²
bandwidth = 30     # in nm
fluence = 1e-3      # in J/cm²
gauss_order = 4

seed_positive = Seed_CPA(bandwidth=bandwidth, fluence=fluence, gauss_order=gauss_order, chirp="positive")
amplifier_positive = Amplifier(crystal=crystal, pump=pump, seed=seed_positive, max_fluence=max_fluence, passes=passes, losses=losses, fast_CPA_computing=False)

seed_negative = Seed_CPA(bandwidth=bandwidth, fluence=fluence, gauss_order=gauss_order, chirp="negative")
amplifier_negative = Amplifier(crystal=crystal, pump=pump, seed=seed_negative, max_fluence=max_fluence, passes=passes, losses=losses, fast_CPA_computing=False)

# for a CPA pulse with a very small bandwidth, the spectral shape and the temporal profile should look similar
spectral_fluence_positive = amplifier_positive.extraction_CPA()
spectral_fluence_negative = amplifier_negative.extraction_CPA()

plot_inversion1D(amplifier_positive)
plot_total_fluence_per_pass(amplifier_positive, save=False, save_path=None)
plt.figure()
plt.plot(seed_positive.lambdas*1e9, spectral_fluence_positive[-1]*1e-9*1e-4, label=f"positive chirp, F = {amplifier_positive.total_fluence_out[-1]*1e-4:.2f} J/cm²")
plt.plot(seed_negative.lambdas*1e9, spectral_fluence_negative[-1]*1e-9*1e-4, label=f"negative chirp, F = {amplifier_negative.total_fluence_out[-1]*1e-4:.2f} J/cm²")
plt.xlabel("wavelength in nm")
plt.ylabel("spectral fluence in J/cm²/nm")
plt.title("Spectral fluence of CPA pulse, positive and negative chirp")
plt.legend()
