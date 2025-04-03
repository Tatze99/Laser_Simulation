import numpy as np
from matplotlib import pyplot as plt
from LaserSim.crystal import Crystal 
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.amplifier import Amplifier, plot_total_fluence_per_pass, plot_temporal_fluence, plot_spectral_fluence

crystal = Crystal(material="YbCaF2", smooth_sigmas=True)
pump = Pump(intensity=40, duration=4)

fluence = 1e-4      # in J/cm²
gauss_order = 4
passes = 80
losses = 0.078
bandwidth = 0.1     # in nm
chirp = "negative"  # "positive", "negative"
max_fluence = 10    # in J/cm²

seed = Seed(gauss_order=gauss_order, fluence=fluence)
amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed, max_fluence=max_fluence, passes=passes, losses=losses)

seed_CPA = Seed_CPA(bandwidth=bandwidth, fluence=fluence, gauss_order=gauss_order, chirp=chirp)
amplifier_CPA = Amplifier(crystal=crystal, pump=pump, seed=seed_CPA, max_fluence=max_fluence, passes=passes, losses=losses)

# compare the total output fluence between the CW pulse and the CPA pulse
plot_total_fluence_per_pass(amplifier, save=False, save_path=None)
amplifier_CPA.extraction_CPA()
plt.plot(amplifier_CPA.total_fluence_out*1e-4, label="output fluence CPA")
plt.legend()

# for a CPA pulse with a very small bandwidth, the spectral shape and the temporal profile should look similar
plot_temporal_fluence(amplifier, save=False, save_path=None)
plot_spectral_fluence(amplifier_CPA, save=False, save_path=None)


