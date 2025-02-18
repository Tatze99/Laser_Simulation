from utilities import numres, h, c, integ
import numpy as np
import os
from glob import glob
import json
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)

Folder = os.path.dirname(os.path.abspath(__file__))


class Spectral_Losses():
    def __init__(self, material= "YbCaF2"):
        File = os.path.join(Folder, "material_database", "reflectivity_curves", material)
        self.load_basedata(os.path.join(File, material + "_Metadata.json"))
        self.fnames = glob(os.path.join(File, self.name +"*"+ self.file_type))
        self.arrays = np.array([np.loadtxt(f)[:,1] for f in self.fnames])*self.reflectivity_unit
        self.lambdas = np.loadtxt(self.fnames[0])[:,0]*self.spectral_unit
        self.dlambda = self.lambdas[1] - self.lambdas[0]
        self.reflectivity = self.arrays[1]
        self.calc_angle_formula()


    def calc_max_index(self, array):
        index = np.argmin(array)
        for j in range(2, len(array)-2):
            if array[j] >= array[j-1] and array[j] >= array[j+1]:
                print(j, array[j])
                if array[j] > array[index]:
                    index =  j
        return index 
    
    def load_basedata(self, filename):
        with open(filename, "r") as file:
            self.dict = json.load(file)
            self.name = self.dict["name"]
            self.angles = self.dict["angles"]
            self.file_type = self.dict["file_type"]
            self.reflectivity_unit = self.dict["reflectivity_unit"]
            self.spectral_unit = self.dict["spectral_unit"]

    def calc_angle_formula(self):                
        L_max = np.zeros(len(self.arrays))
        R_max = np.zeros(len(self.arrays))
        for i, array in enumerate(self.arrays):
            index = self.calc_max_index(array)
        
            L_max[i] = self.lambdas[index]
            R_max[i] = array[index]
            
            # print(self.angles[i], L_max[i], R_max[i])
        
        phi = np.pi/180*np.array(self.angles)
        
        # use first and last curve to determine the shift in wavelength for a given angle
        self.n2 = np.sqrt(((L_max[0]*np.sin(phi[-1]))**2-(L_max[-1]*np.sin(phi[0]))**2)/(L_max[0]**2-L_max[-1]**2))
        self.prop_constant = L_max[0]/np.sqrt(self.n2**2-np.sin(phi[0])**2)
        self.slope = (1 - R_max[0]/R_max[-1])/(L_max[-1] - L_max[0])
    

    def calc_reflectivity(self, phi, angle_unit="rad"):
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
        # print(L_max, int(Delta_lambda/self.dlambda))
        
        
        return shifted_array

if __name__ == "__main__":
    losses = Spectral_Losses(material="YbFP15")

    plt.figure()
    plt.tick_params(direction="in",right=True,top=True)

    # plot the originally measured spectra
    for i,f in enumerate(losses.fnames):
        plt.plot(losses.lambdas*1e9, losses.arrays[i],label=f[-5-len(str(losses.angles[i])):-4])

    # reset color cycle
    plt.gca().set_prop_cycle(None)
    # plot the wavelength shifted spectra determined from the 45° measurement
    for i,angle in enumerate(losses.angles):
        name = losses.fnames[i][-5-len(str(losses.angles[i])):-4]
        plt.plot(losses.lambdas*1e9, losses.calc_reflectivity(angle, angle_unit="deg"), "--", label=name)

    # plt.ylim(0,)
    plt.xlim(1000,1080)
    plt.xlabel("wavelength in nm")
    plt.ylabel("reflectivity R")
    plt.legend(loc="upper right")
    plt.title("Reflectivity TSF of "+ losses.name)

    # Serialize data into file:
    # with open(os.path.join(Folder, material["name"]+'_Metadata.json'), 'w', encoding='utf-8') as f: 
    #     json.dump(material, f, ensure_ascii=False, indent=4)