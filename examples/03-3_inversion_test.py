# Import the pump, crystal, seed and amplifier
from LaserSim.pump import Pump
from LaserSim.crystal import Crystal
from LaserSim.seed import Seed
from LaserSim.amplifier import Amplifier, plot_inversion1D, plot_inversion2D

# In the following we load all the necessary components for an amplifier simulation explicitly.
# This is no different from just calling: amplifier = Amplifier(), as the default values are loaded.


# Load the crystal with its standard values
crystal = Crystal(material="YbLiMgAS_Tiegel", # material of the crystal
                  temperature = 300,  # temperature of the crystal in K
                  length = 2.4e-2,
                  tau_f = 1000,
                  resolution=2000
                ) 


# Load the pump with its standard values
pump = Pump(intensity = 60,         # input intensity in kW/cm²
            duration = 1,           # pulse duration in ms
            wavelength = 940,       # monochromatic wavelength of the quasi-CW pulse
            ) 

amplifier = Amplifier(crystal, pump, symmetric_pump=False)

plot_inversion1D(amplifier)

# plot_inversion2D(amplifier)

print(amplifier.pump.absorbed_energy)