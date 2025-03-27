from LaserSim.seed import Seed, plot_seed_pulse

# Load the CW laser seed with its standard values
seed = Seed(fluence = 0.01,         # input fluence in J/cm²
            duration = 5,           # pulse duration in ns
            wavelength = 1030,      # monochromatic wavelength of the pulse
            gauss_order = 1,        # Gaussian order if seed_type = "gauss" is chosen
            seed_type = "gauss")    # seed shape type: "gauss", "rect" or "lorentz"

# print all information about the crystal (length, tau_f, N_dop, cross sections)
print(seed)

# plot the temporal seed pulse shape, if "save_path = None", the plots will be generated in "../material_database/plots"
plot_seed_pulse(seed)


