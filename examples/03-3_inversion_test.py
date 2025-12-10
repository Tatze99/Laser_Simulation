# Import the pump, crystal, seed and amplifier
from LaserSim.pump import Pump
from LaserSim.crystal import Crystal
from LaserSim.seed import Seed
from LaserSim.amplifier import Amplifier, plot_inversion1D, plot_inversion2D

# In the following we load all the necessary components for an amplifier simulation explicitly.
# This is no different from just calling: pump = Pump() or crystal = Crystal(), as the default values are loaded.


# Load the crystal with its standard values
crystal = Crystal(material="YbCaF2", # material of the crystal
                  temperature = 300,  # temperature of the crystal in K
                ) 


# Load the pump with its standard values
pump = Pump(intensity = 30,         # input intensity in kW/cm²
            duration = 2,           # pulse duration in ms
            wavelength = 940,       # monochromatic wavelength of the quasi-CW pulse
            symmetric = False
            ) 

pump.symmetric = False # standard value is False
amplifier = Amplifier(crystal, pump, print_iteration=False)
plot_inversion1D(amplifier)

print(f"absorbed energy: {amplifier.pump.absorbed_energy[-1]:.5f} (asymmetric pump)")

pump.symmetric = True
amplifier = Amplifier(crystal, pump, print_iteration=False)
plot_inversion1D(amplifier)

print(f"absorbed energy: {amplifier.pump.absorbed_energy[-1]:.5f} (symmetric pump)")