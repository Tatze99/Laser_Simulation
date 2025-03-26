# LaserSim

LaserSim is a python based library which was written to simulate the temporal and spectral profile of a front-pumped CPA laser(diode) amplifier. A range of solid state laser material can be chosen for the simulation. It is possible to simulate and display the inversion inside the laser material during the pump process, show the temporal or spectral fluence during amplification of a given seed pulse and incorporate spectral and constant losses into the cavity.

### Authors: 
- Martin Beyer m.beyer@uni-jena.de
- Clemens Anschütz clemens.anschuetz@uni-jena.de


The main functions are:
1. __Crystal__: material, temperature -> display: cross sections, equilibrium inversion, pump saturation intensity, small signal gain
2. __Pump__: define the (CW) pump pulse: pump intensity, duration, absorption wavelength
3. __Seed__: define the temporal seed profile: duration, wavelength, fluence, shape (Gaussian, Lorentzian, Rectangular)
4. __Seed_CPA__: define the spectral seed profile: central wavelength, bandwidth, fluence, shape (Gaussian, Rectangular)
5. __Amplifier__:
    - Calculate the inversion numerically by solving the coupled rate equation of the pump rate $R(z,t)$ and the inversion $\beta(z,t)$. We also consider spontaneous emission as a loss mechanism
    - Calculate the extracted fluence and the inversion after extraction using the analytical solution of the Frantz-Nodvik equation solved for a 3-level system (with absorption at the pump wavelength). 

## How to install LaserSim:
- Install Python 3.12 (recommended)
- Required Packages: numpy, matplotlib, scipy
- Download the repository to an arbitrary location
```
git clone https://github.com/tatze99/LaserSimulation.git
```

## How to setup your own script

Paste the following lines at the beginning of a new python file: 
```
import os
import sys
sys.path.append(os.path.abspath("PROJECT_PATH"))
Folder = os.path.dirname(os.path.abspath(__file__))
```

```
your_project/
│── your_library/         # Main project (library code)
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│── simulations/          # Your separate simulation scripts
│   ├── my_experiment.py  # Your own simulations
│── tests/                # Unit tests for the library
│── examples/             # Example scripts for users
│── docs/                 # Documentation
│── requirements.txt
│── README.md
```
