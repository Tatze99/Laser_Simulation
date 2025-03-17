# LaserSim
Authors: 
- Martin Beyer m.beyer@uni-jena.de
- Clemens Anschütz clemens.anschuetz@uni-jena.de

In short: LaserSim is a program to simulate the temporal and spectral profile of a front-pumped CPA laser(diode) amplifier. You can simulate the pumping of the laser active material for various solid state laser materials and from that calculate the output fluence.

The main functions are:
1. Crystal: material, temperature -> display: cross sections, equilibrium inversion, pump saturation intensity, small signal gain
2. __Pump__: define the (CW) pump pulse: pump intensity, duration, absorption wavelength
3. __Seed__: define the temporal seed profile: duration, wavelength, fluence, shape (Gaussian, Lorentzian, Rectangular)
4. __Seed_CPA__: define the spectral seed profile: central wavelength, bandwidth, fluence, shape (Gaussian, Rectangular)
5. __Amplifier__:
    - Calculate the inversion numerically by solving the coupled rate equation of the pump rate $R(z,t)$ and the inversion $\beta(z,t)$. We also consider spontaneous emission as a loss mechanism
    - Calculate the extracted fluence and the inversion after extraction using the analytical solution of the Frantz-Nodvik equation solved for a 3-level system (with absorption at the pump wavelength). 

# How to use:
- Install Python 3.x
- Required Packages: numpy, matplotlib, scipy
- Download the repository to an arbitrary location
```
git clone https://github.com/tatze99/LaserSimulation.git
```