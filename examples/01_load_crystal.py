# Import the Crystal
from LaserSim.crystal import Crystal, plot_small_signal_gain, plot_cross_sections, plot_Fsat

# Load the laser Crystal, the length and doping concentration are read from the database, but can also be given manually using "length" and "N_dop"

crystal = Crystal(material = "YbCaF2",
                  temperature = 300,
                  length = None,
                  N_dop = None, 
                  smooth_sigmas=True)

# print all information about the crystal (length, tau_f, N_dop, cross sections)
print(crystal)

# plot the cross sections, one can specify if the plot should be saved, if "save_path = None", the plots will be generated in "../material_database/plots"
plot_cross_sections(crystal)

# define a inversion at which we want to plot the small signal gain for the length specified in the crystal class
beta = 0.3
# plot the small signal gain
plot_small_signal_gain(crystal, beta, save=False, save_path=None)

plot_Fsat(crystal)

