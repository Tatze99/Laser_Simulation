# Import the pump, crystal, seed and amplifier
from LaserSim.pump import Pump
from LaserSim.crystal import Crystal
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.amplifier import Amplifier, plot_spectral_fluence

# In the following we load all the necessary components for an amplifier simulation explicitly.


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
seed = Seed_CPA(wavelength = 1030,      # central wavelength of the pulse
                bandwidth = 30,         # bandwidth of the pulse
                fluence = 01e-4,        # input fluence in J/cm²
                gauss_order = 1,        # Gaussian order if seed_type = "gauss" is chosen
                seed_type = "gauss",    # seed shape type: "gauss", "rect" or "lorentz"
                custom_file = None)     # custom file for seed pulse

# Load the amplifier with its standard values
amplifier = Amplifier(crystal, pump, seed, 
                      passes = 50,
                      losses = 0.02,
                      max_fluence = 10) 

# print all information about the amplifier (pump, seed, crystal)
print(amplifier)

# plot the temporal seed pulse shape, if "save_path = None", the plots will be generated in "../material_database/plots"
plot_spectral_fluence(amplifier)