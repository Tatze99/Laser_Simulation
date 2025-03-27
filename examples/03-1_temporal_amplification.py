# Import the pump, crystal, seed and amplifier
from LaserSim.pump import Pump
from LaserSim.crystal import Crystal
from LaserSim.seed import Seed
from LaserSim.amplifier import Amplifier, plot_temporal_fluence, plot_total_fluence_per_pass

# In the following we load all the necessary components for an amplifier simulation explicitly.
# This is no different from just calling: amplifier = Amplifier(), as the default values are loaded.


# Load the crystal with its standard values
crystal = Crystal(material="YbCaF2", # material of the crystal
                  temperature = 300,  # temperature of the crystal in K
                ) 

# Load the pump with its standard values
pump = Pump(intensity = 30,         # input intensity in kW/cm²
            duration = 2,           # pulse duration in ns
            wavelength = 940,       # monochromatic wavelength of the quasi-CW pulse
            ) 

# Load the CW laser seed with its standard values
seed = Seed(fluence = 0.01,         # input fluence in J/cm²
            duration = 5,           # pulse duration in ns
            wavelength = 1030,      # monochromatic wavelength of the pulse
            gauss_order = 1,        # Gaussian order if seed_type = "gauss" is chosen
            seed_type = "gauss")    # seed shape type: "gauss", "rect" or "lorentz"

# Load the amplifier with its standard values
amplifier = Amplifier(crystal, pump, seed, 
                      passes = 50,
                      losses = 0.02,
                      max_fluence = 10)

# print all information about the amplifier (pump, seed, crystal)
print(amplifier)

# plot the temporal seed pulse shape, if "save_path = None", the plots will be generated in "../material_database/plots"
plot_temporal_fluence(amplifier)

plot_total_fluence_per_pass(amplifier) 