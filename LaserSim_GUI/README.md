# LaserSim_GUI
Graphical user interface for LaserSim. It acts as a *submodule* of Laser_Simulation and cannot be used as a standalone.
See also **https://github.com/Tatze99/Laser_Simulation**

The program accesses functions written in Laser_Simulation and displays them in a ```customtkinter``` interface (c.f. package requirements). The interface includes many entries to adjust the laser parameters in an easy way. It allows for multiplotting. You can add your own spectroscopic data by navigating to the ```material_database``` folder in the parent directory.

### Author: 
- Martin Beyer m.beyer@uni-jena.de

## Features
In the following, we want to show different uses and features of the graphical user interface. There are two ```Plot``` buttons on the left of the interface. The list above the plot button selects the function you want to display. Some functions have extra arguments that will appear/disappear when selecting the specific function. 
- The first button is concerned with spectroscopic properties of the material without ever having to specify and pump or a seed pulse. It only invokes a ```Crystal()``` object. You can display the cross sections, saturation intensity and fluence, the equilibrium inversion etc.
- The second button creates a ```Pump()```, ```Seed()``` and ```Amplifier()``` object and uses them to simulate amplification, inversion etc.

### Display spectroscopic data of the materials in the database of Laser_Simulation
- plot the absorption and emission cross sections for different temperatures
- adjust the doping concentration, thickness, fluorescence lifetime and position of the zero phonon line (ZPL)
- plot the saturation intensity and saturation fluence for different wavelengths
- plot the small signal gain for a set of inversions
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_cross_sections.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_small_signal_gain.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
</p>


### Simulation of the pump process
- adjust the pump parameters: pump duration, intensity and wavelength
- display the temporal and spatial evolution of the inversion inside the laser material
- display the storage efficiency as a function of pump duration and/or pump intensity
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_inversion_spatial.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_storage_efficiency.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
</p>

### Simulation of amplification in a resonator
- configure the seed settings: puls duration, initial fluence, central wavelength, bandwidth, pulse shape (gaussian, lorentzian, rectangular)
- configure the amplifier settings: number of passes in through the medium, losses per round trip in the cavity, maximum fluence before the simulation stops
- calculate and display the total fluence of the seed pulse after every pass through the medium
- display the temporal pulse profile for every pass -> show effects like edge steepening
- display the spectral pulse profile for every pass -> show effects like spectral gain narrowing

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_total_fluence_per_pass.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_spectral_fluence.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
</p>

## Special features
- It is possible to enable **multi-plotting**: Clicking the button creates a new button which can be used to reset the plotting area. The "Plot" button behaviour changes such as the plot is not re-created, but a new line is added. This can be used to compare simulation outputs for different amplifier settings or to plot cross sections on top of each other.
- for some plots, the **legend can be customized** and replaced by your own legend (this feature can be seen in the image showing the output fluence vs. pass number)
- Here, the simulation parameters can be accessed using specific keywords written in curly brackets e.g.: ```I = {intensity} kW/cm²```

Here is a list of these keywords:
|keyword|meaning|unit|
|---|---|---|
|crystal|material name||
|intensity|pump intensity|kW/cm²|
|tau_p|pump duration|ms|
|Ndop|doping concentration|1/cm³|
|thickness|crystal thickness|mm|
|temperature|crystal temperature|K|
|ZPL|crystal zero phonon line|nm|
|tau_f|crystal fluorescence lifetime|ms|
|losses|cavity losses|%|
|lambda_p|pump wavelength|nm|
|lambda_l|(Q-Switch) laser wavelength|nm|

- You can also specify a **working directory** and save your current project into a ```.json``` file. This saves all current entries and can be reloaded when necessary.
- You can customize the plot by showing/hiding the grid and by showing/hiding the plot title
- The size of the plot and aspect ratio can be selected to save the plot as .pdf or image file
- The current plot data (all lines) can be saved into a .txt file when specifying this as the file type when saving.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./documentation_images/GUI_settings.png">
    <img src="./documentation_images/Program_start.png">
  </picture>
</p>

## Installation
- Install Laser_Simulation (https://github.com/Tatze99/Laser_Simulation)
- change the directory to the Laser_Simulation folder
- Clone this repository into the Laser_Simulation folder
```
git clone https://github.com/Tatze99/LaserSim_GUI.git
```
- install customtkinter to run the GUI
```
pip install customtkinter
```
- run the ```LaserSim_GUI.py``` file
