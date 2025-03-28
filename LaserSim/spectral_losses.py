from LaserSim.utilities import set_plot_params
import numpy as np
import os
from glob import glob
import json
import matplotlib.pyplot as plt
set_plot_params()

Folder = os.path.dirname(os.path.abspath(__file__))
Folder = os.path.abspath(os.path.join(Folder, os.pardir))

class Spectral_Losses():
    def __init__(self, material= "YbFP15", calc_formula = True, custom_File = None):
        File = os.path.join(Folder, "material_database", "reflectivity_curves", material) if custom_File is None else custom_File
        self.TSF_name = material
        self.load_basedata(os.path.join(File, "*_Metadata.json"))
        self.fnames = glob(os.path.join(File, "*"+ self.file_type))
        self.arrays = np.array([np.loadtxt(f)[:,1] for f in self.fnames])*self.reflectivity_unit
        self.lambdas = np.loadtxt(self.fnames[0])[:,0]*self.spectral_unit
        self.dlambda = self.lambdas[1] - self.lambdas[0]
        if calc_formula and len(self.angles) > 1:
            self.reflectivity = self.arrays[1]
            self.calc_angle_formula()
        else:
            print("loading angle formula from metadata file") 
            self.reflectivity = self.arrays[0]
            self.load_angle_formula()

    # calculate the index of the maximum value of an array
    def calc_max_index(self, array):
        index = np.argmin(array)
        for j in range(2, len(array)-2):
            if array[j] >= array[j-1] and array[j] >= array[j+1]:
                if array[j] > array[index]:
                    index =  j
        return index 
    
    # load the basic data from a json file
    def load_basedata(self, file):
        filename = glob(file)[0]
        with open(filename, "r") as file:
            self.dict = json.load(file)
            self.name = self.dict["name"]
            self.angles = self.dict["angles"]
            self.file_type = self.dict["file_type"]
            self.reflectivity_unit = self.dict["reflectivity_unit"]
            self.spectral_unit = self.dict["spectral_unit"]

    # calculate the angle formula constants (n2, prop_constant) from the three given reflectivity curves,
    def calc_angle_formula(self):                
        L_max = np.zeros(len(self.arrays))
        R_max = np.zeros(len(self.arrays))
        for i, array in enumerate(self.arrays):
            index = self.calc_max_index(array)
        
            L_max[i] = self.lambdas[index]
            R_max[i] = array[index]
        
        phi = np.pi/180*np.array(self.angles)
        
        # use first and last curve to determine the shift in wavelength for a given angle
        self.n2 = np.sqrt(((L_max[0]*np.sin(phi[-1]))**2-(L_max[-1]*np.sin(phi[0]))**2)/(L_max[0]**2-L_max[-1]**2))
        self.prop_constant = L_max[0]/np.sqrt(self.n2**2-np.sin(phi[0])**2)
        self.slope = (1 - R_max[0]/R_max[-1])/(L_max[-1] - L_max[0])
    
    def load_angle_formula(self):
        """
        load the angle formula from the metadata if there are no reflectivity curves given to calculate it
        this is useful, if a different design is tested with similar angle formula constants"""
        self.n2 = self.dict["n2"]
        self.prop_constant = self.dict["prop_constant"]
        self.slope = self.dict["slope"]


    def calc_reflectivity(self, phi, angle_unit="rad"):
        """
        Calculate the reflectivity for a given angle phi in radians or degrees by shifting the original reflectivity curve and multiply it with a linear factor (self.slope)
        """
        # ensure that the reflectivity is not given in percent, and the angle is in radians
        if angle_unit == "deg": phi *= np.pi/180

        L_max = self.lambdas[self.calc_max_index(self.reflectivity)]
        # calculate wavelength maxima for the given angle phi
        L_max_phi = self.prop_constant*np.sqrt(self.n2**2-np.sin(phi)**2)
        
        Delta_lambda = L_max_phi -L_max
        
        # shift the array to the new max_wavelength
        shifted_array = np.roll(self.reflectivity, int(Delta_lambda/self.dlambda))
        # transform the maximum amplitude due to the linear transform
        shifted_array *= (1 + self.slope * Delta_lambda)
        
        return shifted_array
    
    def calc_total_reflectivity(self, reflectivity_array, n=0):
        """
        Calculate the total reflectivity for a given array of reflectivities by recursively multiplying the reflectivity with the complement of the previous reflectivity
        """
        if n == len(reflectivity_array)-1:
            return reflectivity_array[n]
        return reflectivity_array[n] + (1-reflectivity_array[n])*self.calc_total_reflectivity(reflectivity_array, n+1)
    
    def reflectivity_by_angles(self, angle_array, angle_unit="grad"):
        """
        Calculate the total reflectivity for a given array of angles in radians or degrees
        """
        reflectivity_array = []
        if angle_array is None:
            return np.zeros(len(self.lambdas))
        
        for angle in angle_array:
            reflectivity_array.append(self.calc_reflectivity(angle, angle_unit=angle_unit))
        return self.calc_total_reflectivity(np.array(reflectivity_array))

    def __repr__(self):
        return(f"Spectral reflectivity mirrors:\n"
                f"- name: {self.name}\n"
                f"- angles = {self.angles}\n"
                f"- file type = {self.file_type}\n"
                f"- reflectivity_unit = {self.reflectivity_unit}\n"
                f"- spectral_unit = {self.spectral_unit}\n"
                f"- n2 = {self.n2}\n"
                f"- prop_constant = {self.prop_constant}\n"
                f"- slope = {self.slope}\n"
        )

def test_reflectivity_approximation(losses, save=False):
    """
    Test the reflectivity approximation by comparing the calculated reflectivity with the original reflectivity curves
    """
    plt.figure()
    plt.tick_params(direction="in",right=True,top=True)
    colors = plt.cm.tab10.colors

    for i,fname in enumerate(losses.fnames):
        name = fname[-5-len(str(losses.angles[i])):-4]
        plt.plot(losses.lambdas*1e9, losses.arrays[i],label=name, color=colors[i])
        plt.plot(losses.lambdas*1e9, losses.calc_reflectivity(losses.angles[i], angle_unit="deg"), "--", label=name, color=colors[i])

    plt.xlim(1000,1080)
    plt.ylim(0,1e-1)
    plt.xlabel("wavelength in nm")
    plt.ylabel("reflectivity R")
    plt.legend(loc="upper right")
    plt.title("Reflectivity TSF of "+ losses.name)

    if save:
        plt.tight_layout()
        plt.savefig(os.path.join(Folder, "material_database","plots", f"{losses.TSF_name}_reflectivity_approximation.pdf"))

if __name__ == "__main__":
    losses = Spectral_Losses()

    print(losses)
    test_reflectivity_approximation(losses, save=False)