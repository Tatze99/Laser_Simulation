# LaserSim

LaserSim is a python based library which was written to simulate the temporal and spectral profile of a front-pumped CPA laser(diode) amplifier. A range of solid state laser material can be chosen for the simulation. It is possible to simulate and display the inversion inside the laser material during the pump process, show the temporal or spectral fluence during amplification of a given seed pulse and incorporate spectral and constant losses into the cavity.

### Authors: 
- Martin Beyer m.beyer@uni-jena.de
- Clemens Anschütz clemens.anschuetz@uni-jena.de


## Main functions
1. __Crystal__: material, temperature -> display: cross sections, equilibrium inversion, pump saturation intensity, small signal gain
2. __Pump__: define the (CW) pump pulse: pump intensity, duration, absorption wavelength
3. __Seed__: define the temporal seed profile: duration, wavelength, fluence, shape (Gaussian, Lorentzian, Rectangular)
4. __Seed_CPA__: define the spectral seed profile: central wavelength, bandwidth, fluence, shape (Gaussian, Rectangular)
5. __Amplifier__:
    - Calculate the inversion numerically by solving the coupled rate equation of the pump rate $R(z,t)$ and the inversion $\beta(z,t)$. We also consider spontaneous emission as a loss mechanism
    - Calculate the extracted fluence and the inversion after extraction using the analytical solution of the Frantz-Nodvik equation solved for a 3-level system (with absorption at the pump wavelength). 

## Project structure
```
Laser_Simulation/
│── examples/
│   ├── 01_load_crystal.py              # Loading the laser active medium
│   ├── 02_load_seed.py                 # Loading a temporal seed pulse
│   ├── 03-1_temporal_amplification.py  # Simulate an temporal pulse in the amplifier
│   ├── 03-2_spectral_amplification.py  # Simulate a spectrum in the amplifier 
│── LaserSim/                           # Main project (library code)
│   ├── __init__.py                     # Initialize LaserSim as a package
│   ├── amplifier.py
│   ├── crystal.py
│   ├── pump.py
│   ├── seed_CPA.py
│   ├── seed.py
│   ├── spectral_losses.py              # Calculate spectral losses with tunable spectral filters (TSFs)
│   ├── utilities.py                    # Helper functions, plot settings, constants
│── material_database/
│   ├── plots/                          # contains the generated plots, when they are saved
│   ├── reflectivity_curves/            # contains spectral reflectivity curves of the TSFs
│   ├── Ybxxx/                          # Contains data about a specific material
│── requirements.txt                    # Contains the required packages to install with pip
│── setup.py                            # file to pack "LaserSim" into a package
|── tutorial.ipynb                      # Jupyter Notebook with a tutorial how to get started
│── README.md
```

# How to guides

### How to install LaserSim:
- Install Python 3.12 (recommended)
- Download the repository to an arbitrary location
```
git clone https://github.com/tatze99/LaserSimulation.git    # or use the Github Desktop application
```
- open this folder in a python capable IDE, e.g. VisualStudioCode
- Optional: open a Terminal in this folder, create a virtual environment:
```
python -m venv ./venv
```
- This will install a virtual environment in the "venv" folder
- Activate the virtual environment by running activate.bat
```
./venv/Scripts/activate      # on Windows
source ./venv/bin/activate   # on Linux
```
- install the required packages:
```
python -m pip install -r ./requirements.txt
```
- Finally, we need to add the root folder to the system path
```
set PYTHONPATH=%CD%         # on Windows
export PYTHONPATH=$(pwd)    # on Linux
```
- Now you can run the example files or the tutorial.ipynb file to test if everything works

### How to uninstall LaserSim:
- open a Terminal in the project's root folder
```
./venv/scripts/deactivate.bat     
```
- Then you can delete the "venv" and "LaserSim.egg-info" folder

### How to setup your own script

- create a subfolder in the main directory
- Create a new python file and load the needed functions in the following way
```
from LaserSim.seed import Seed
from LaserSim.pump import Pump
```


