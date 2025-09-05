import os
import re
import json
from CTkRangeSlider import *
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from PIL import Image
import ast # for literal_eval of a user input

from LaserSim.crystal import Crystal, plot_cross_sections, plot_small_signal_gain, plot_beta_eq, plot_Isat, plot_Fsat
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.spectral_losses import Spectral_Losses
from LaserSim.amplifier import Amplifier, plot_inversion1D, plot_inversion2D, plot_temporal_fluence, plot_total_fluence_per_pass, plot_spectral_fluence, plot_inversion_before_after, plot_inversion_vs_pump_intensity
from LaserSim.utilities import integ, set_plot_params, numres
from LaserSim.seed import plot_seed_pulse as plot_QSwitch_pulse
from LaserSim.seed_CPA import plot_seed_pulse as plot_CPA_pulse

from LaserSim.spectral_losses import test_reflectivity_approximation


version_number = "25/09"
Standard_path = os.path.dirname(os.path.abspath(__file__))
LaserSim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(LaserSim_path)

plt.style.use('default')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        set_plot_params()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Laser Sim."+version_number)
        self.geometry("1280x720")
        self.replot = False
        self.initialize_variables()
        self.initialize_ui_images()
        self.initialize_ui()

        self.toplevel_window = {'Plot Settings': None,
                                'Legend Settings': None}

        self.setup_plot_area()
        self.update_material(self.material_list.get())
        self.crystal_plot()

        self.save_attributes = ["material_list", "temperature_list", "material_plot_list", "amplifier_plot_list", "folder_path"]

        self.save_attributes.extend(self.crystal_widgets)
        self.save_attributes.extend(self.seed_widgets)
        self.save_attributes.extend(self.pump_widgets)
        self.save_attributes.extend(self.amplifier_widgets)

    def initialize_ui_images(self):
        self.img_settings = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","options.png")), size=(15, 15))
        self.img_save = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","save_white.png")), size=(15, 15))
        self.img_folder = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","folder.png")), size=(15, 15))
        self.img_reset = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","reset.png")), size=(15, 15))
        self.img_next = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","next_arrow.png")), size=(15, 15))
        self.img_previous = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","previous_arrow.png")), size=(15, 15))

    def initialize_variables(self):
        folder = os.path.join(LaserSim_path, "material_database")
        self.materials = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and not f.startswith('plots') and not f.startswith('reflectivity_curves')]
        self.material_plot_functions = {'Cross sections': plot_cross_sections, 
                                        'Small signal gain': plot_small_signal_gain,
                                        'Equilibrium inversion': plot_beta_eq, 
                                        'Saturation intensity': plot_Isat, 
                                        'Saturation fluence': plot_Fsat
                                        }
        
        self.amplifier_plot_functions = {'Temporal fluence': plot_temporal_fluence, 
                                        'Total fluence pass': plot_total_fluence_per_pass,
                                        'Spectral fluence': plot_spectral_fluence,
                                        'Inversion 1D': plot_inversion1D, 
                                        'Inversion 2D': plot_inversion2D,
                                        'Inversion vs Ip': plot_inversion_vs_pump_intensity, 
                                        'Inversion before vs after': plot_inversion_before_after
                                        }

        self.ax = None
        self.plot_index = 0
        self.color = "#212121" # toolbar
        self.text_color = "white"

        # line plot settings
        self.moving_average = 1
        
        # Boolean variables
        self.save_plain_image = customtkinter.BooleanVar(value=False)
        self.seed_CPA = customtkinter.BooleanVar(value=False)

    # user interface, gets called when the program starts 
    def initialize_ui(self):

        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, height=600, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nesw")
        self.sidebar_frame.rowconfigure(15, weight=1)
        self.rowconfigure(11,weight=1)

        App.create_label(self.sidebar_frame, text="LaserSim v."+version_number, font=customtkinter.CTkFont(size=20, weight="bold"),row=0, column=0, padx=20, pady=(10,15), sticky=None)
        
        self.tabview = customtkinter.CTkTabview(self, width=250, command=lambda: self.toggle_toolbar(self.tabview.get()))
        self.tabview.grid(row=3, column=1, padx=(10, 10), pady=(20, 0), columnspan=2, sticky="nsew", rowspan=10)
        self.tabview.add("Show Plots")
        self.tabview.add("Data Table")
        self.tabview.tab("Show Plots").columnconfigure(0, weight=1)
        self.tabview.tab("Show Plots").rowconfigure(0, weight=1)

        #buttons
        
        frame = self.sidebar_frame

        self.material_list  = App.create_Menu(frame, values=self.materials, column=0, row=1, command=self.update_material, width=135, padx = (10,210-135), init_val=self.materials[1])
        self.temperature_list  = App.create_Menu(frame, values=["300"], column=0, row=1, command=None, width=60, padx = (210-60,10))

        self.material_plot_list  = App.create_Menu(frame, values=list(self.material_plot_functions.keys()), column=0, row=2, pady=(15,5), command=self.toggle_extra_material_arguments)
        self.plot_crystal_button    = App.create_button(frame, text="Plot material", command=self.crystal_plot, column=0, row=4, image=self.img_reset)
        
        # extra settings
        self.inversion    = App.create_entry(frame, column=0, row=5, init_val="0.1,0.15,0.2", width=170, padx=(210-170, 10))
        self.inversion_label    = App.create_label(frame, column=0, row=5, text="β", padx=(5, 215-20))
        self.plot_pump_laser_cross_sections = App.create_switch(frame, text="Show σ(λ_p), σ(λ_l)", command=None,  column=0, row=5, padx=20)


        self.amplifier_plot_list  = App.create_Menu(frame, values=list(self.amplifier_plot_functions.keys()), column=0, row=7, pady=(25,5), command=None)
        self.plot_amplifier_button = App.create_button(frame, text="Plot amplifier", command=self.amplifier_plot, column=0, row=8, pady=(5,15), image=self.img_reset)

        # self.set_button     = App.create_button(frame, text="Plot settings",command=None, column=0, row=16, image=self.img_settings)
        self.save_data_switch = App.create_switch(frame, text="Save data during plot", command=None,  column=0, row=16, padx=20, pady=(10,5))
        self.save_plot_switch = App.create_switch(frame, text="Save figure during plot", command=None,  column=0, row=17, padx=20)
        self.save_button    = App.create_button(frame, text="Save current figure", command=self.save_figure,     column=0, row=18,  image=self.img_save, pady=(5,15))
        
        #switches
        # self.multiplot_button = App.create_switch(frame, text="Multiplot",  command=None,   column=0, row=7, padx=20, pady=(10,5))
        self.crystal_button      = App.create_switch(frame, text="Config Crystal", command=lambda: self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets),  column=0, row=10, padx=20)
        self.pump_button      = App.create_switch(frame, text="Config Pump", command=lambda: self.toggle_sidebar_window(self.pump_button, self.pump_widgets),  column=0, row=11, padx=20)
        self.seed_button      = App.create_switch(frame, text="Config Seed", command=lambda: self.toggle_sidebar_window(self.seed_button, self.seed_widgets),  column=0, row=12, padx=20)
        self.amplifier_button = App.create_switch(frame, text="Config Amplifier", command=lambda: self.toggle_sidebar_window(self.amplifier_button, self.amplifier_widgets),  column=0, row=13, padx=20)

        #Data Table section
        self.load_button    = App.create_button(self.tabview.tab("Data Table"), text="Set Working directory", command=self.read_file_list,  column=0, row=0, image=self.img_folder, width=230)
        self.folder_path = App.create_entry(self.tabview.tab("Data Table"), row=0, column=2, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        self.json_path = App.create_entry(self.tabview.tab("Data Table"), row=1, column=2, text="project file name", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        self.folder_path.insert(0, Standard_path)
        self.json_path.insert(0, "project_data")
        self.save_data_button    = App.create_button(self.tabview.tab("Data Table"), text="save project", command=lambda: self.save_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")),  column=0, row=1, image=self.img_save, width=110, padx=(10,10+230-110))
        self.load_data_button    = App.create_button(self.tabview.tab("Data Table"), text="load project", command=lambda: self.load_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")),  column=0, row=1, image=self.img_folder, width=110, padx=(10+230-110,10))
        

        self.inversion.grid_remove()
        self.inversion_label.grid_remove()
        self.plot_pump_laser_cross_sections.grid_remove()

        self.load_settings_frame()

    # initialize all widgets on the settings frame
    def load_settings_frame(self):
        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.settings_frame.grid_columnconfigure(0, minsize=60)
        self.columnconfigure(2,weight=1)

        self.load_crystal_sidebar()
        self.load_pump_sidebar()
        self.load_seed_sidebar()
        self.load_amplifier_sidebar()


    def create_label(self, row, column, width=20, text=None, anchor='e', sticky='e', textvariable=None, padx=(5,5), image=None, pady=None, font=None, columnspan=1, fg_color=None, **kwargs):
        label = customtkinter.CTkLabel(self, text=text, textvariable=textvariable, width=width, image=image, anchor=anchor, font=font, fg_color=fg_color,  **kwargs)
        label.grid(row=row, column=column, sticky=sticky, columnspan=columnspan, padx=padx, pady=pady, **kwargs)
        return label

    def create_entry(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, placeholder_text=None, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        entry = CustomEntry(self, width, height, placeholder_text=placeholder_text, **kwargs)
        entry.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            entry.insert(0,str(init_val))
        if text is not None:
            entry_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (entry, entry_label)

        return entry

    def create_button(self, command, row, column, text=None, image=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkButton(self, text=text, command=command, width=width, image=image, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button
    
    def create_segmented_button(self, values, command, row, column, width=200, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        button = customtkinter.CTkSegmentedButton(self, values=values, command=command, width=width, **kwargs)
        button.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return button

    def create_switch(self, command, row, column, text, columnspan=1, padx=10, pady=5, sticky='w', **kwargs):
        switch = customtkinter.CTkSwitch(self, text=text, command=command, **kwargs)
        switch.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        return switch
    
    def create_combobox(self, values, column, row, width=200, state='readonly', command=None, text=None, columnspan=1, padx=10, pady=5, sticky=None, **kwargs):
        combobox = customtkinter.CTkComboBox(self, values=values, command=command, state=state, width=width, **kwargs)
        combobox.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            App.create_label(self, text=text, column=column-1, row=row, width=80, anchor='e',pady=pady)
        return combobox
    
    def create_Menu(self, values, column, row, command=None, text=None, width=200, columnspan=1, padx=10, pady=5, sticky=None, textwidget=False, init_val=None, **kwargs):
        optionmenu = customtkinter.CTkOptionMenu(self, values=values, width=width, command=command, **kwargs)
        optionmenu.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            optionmenu.set(init_val)
        if text is not None:
            optionmenu_label = App.create_label(self, text=text, column=column-1, row=row, anchor='e', pady=pady)
            if textwidget == True:
                return (optionmenu, optionmenu_label)

        return optionmenu
    
    def create_slider(self, from_, to, row, column, width=200, text=None, init_val=None, command=None, columnspan=1, number_of_steps=1000, padx = 10, pady=5, sticky='w', textwidget=False,**kwargs):
        slider = customtkinter.CTkSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            slider_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')
            if textwidget == True:
                return (slider, slider_label)
        if init_val is not None:
            slider.set(init_val)

        return slider
    
    def create_range_slider(self, from_, to, row, column, width=200, text=None, init_value=None, command=None, columnspan=1, number_of_steps=None, padx = 10, pady=5, sticky='w', textwidget=False,**kwargs):
        slider = CTkRangeSlider(self, from_=from_, to=to, width=width, command=command, number_of_steps=number_of_steps)
        slider.grid(column=column, row=row, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if text is not None:
            slider_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e')
            if textwidget == True:
                return (slider, slider_label)
        if init_value is not None:
            slider.set(init_value)

        return slider
    
    def create_table(self,  width, row, column, sticky=None, rowspan=1, **kwargs):
        text_widget = customtkinter.CTkTextbox(self, width = width, padx=10, pady=5)
        # text_widget.pack(fill="y", expand=True)
        text_widget.grid(row=row, column=column, sticky=sticky, rowspan=rowspan, **kwargs)
        self.grid_rowconfigure(row+rowspan-1, weight=1)

        return text_widget
    
    def create_textbox(self, row, column, width=200, height=28, text=None, columnspan=1, rowspan=1, padx=10, pady=5, sticky='w', sticky_label='e', textwidget=False, init_val=None, **kwargs):
        textbox = customtkinter.CTkTextbox(self, width, height, **kwargs)
        textbox.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_val is not None:
            textbox.insert(0,str(init_val))
        if text is not None:
            textbox_label = App.create_label(self, text=text, column=column-1, row=row, width=20, anchor='e', sticky=sticky_label)
            if textwidget == True:
                return (textbox, textbox_label)

        return textbox
    
    # Load the sidebar
    def load_crystal_sidebar(self):
        row = 0
        self.crystal_title = App.create_label(self.settings_frame, text="Crystal Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.crystal_doping, self.crystal_doping_label = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="doping [cm⁻³]", textwidget=True)
        self.crystal_thickness, self.crystal_thickness_label = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="thickness [mm]", textwidget=True)
        self.crystal_tau_f, self.crystal_tau_f_label = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="lifetime [ms]", textwidget=True)

        self.crystal_widgets = ["crystal_title", "crystal_doping", "crystal_thickness", "crystal_tau_f", "crystal_doping_label", "crystal_thickness_label", "crystal_tau_f_label"]
        self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets)

    def load_pump_sidebar(self):
        row = 10
        self.pump_title = App.create_label(self.settings_frame, text="Pump Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.pump_intensity, self.pump_intensity_label = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="intensity [kW/cm²]", init_val=Pump().intensity*1e-7, textwidget=True)
        self.pump_wavelength, self.pump_wavelength_label = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="wavelength [nm]", init_val=round(Pump().wavelength*1e9, 3), textwidget=True)
        self.pump_duration, self.pump_duration_label = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="duration [ms]", init_val=Pump().duration*1e3, textwidget=True)

        self.pump_widgets= ["pump_intensity","pump_intensity_label","pump_wavelength","pump_wavelength_label","pump_title", "pump_duration", "pump_duration_label"]
        self.toggle_sidebar_window(self.pump_button, self.pump_widgets)

    def load_seed_sidebar(self):
        row = 20
        self.seed_title = App.create_label(self.settings_frame, text="Seed Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.seed_type_button = App.create_segmented_button(self.settings_frame, values=["Q-Switch","CPA"], command=lambda value: self.toggle_seed_type(value), row=row+1, column=0, columnspan=3, padx=20, pady=(5, 15), width=200)

        self.seed_QSwitch_fluence, self.seed_QSwitch_fluence_label = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="fluence [J/cm²]", init_val=Seed().fluence*1e-4, textwidget=True)
        self.seed_QSwitch_wavelength, self.seed_QSwitch_wavelength_label = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="wavelength [nm]", init_val=Seed().wavelength*1e9, textwidget=True)
        self.seed_QSwitch_duration, self.seed_QSwitch_duration_label = App.create_entry(self.settings_frame,column=1, row=row+4, columnspan=2, width=110, text="duration [ns]", init_val=Seed().duration*1e9, textwidget=True)
        self.seed_QSwitch_pulsetype, self.seed_QSwitch_pulsetype_label = App.create_Menu(self.settings_frame, values=["gauss","lorentz","rect"], column=1, row=row+5, command=None, text="pulse type", width=110, textwidget=True)

        self.seed_CPA_fluence, self.seed_CPA_fluence_label = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="fluence [J/cm²]", init_val=Seed_CPA().fluence*1e-4, textwidget=True)
        self.seed_CPA_wavelength, self.seed_CPA_wavelength_label = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="wavelength [nm]", init_val=round(Seed_CPA().wavelength*1e9, 3), textwidget=True)
        self.seed_CPA_duration, self.seed_CPA_duration_label = App.create_entry(self.settings_frame,column=1, row=row+4, columnspan=2, width=110, text="bandwidth [nm]", init_val=round(Seed_CPA().bandwidth*1e9, 3), textwidget=True)
        self.seed_CPA_pulsetype, self.seed_CPA_pulsetype_label = App.create_Menu(self.settings_frame, values=["gauss","lorentz","rect"], column=1, row=row+5, command=None, text="pulse type", width=110, textwidget=True)

        self.plot_seed_button = App.create_button(self.settings_frame, text="Plot Seed Pulse", command=self.seed_plot, row=row+6, column=0, columnspan=5, padx=20, pady=(5, 15))

        self.seed_widgets= ["seed_QSwitch_fluence","seed_QSwitch_fluence_label","seed_QSwitch_wavelength","seed_QSwitch_wavelength_label","seed_title", "seed_QSwitch_duration", "seed_QSwitch_duration_label", "seed_QSwitch_pulsetype", "seed_QSwitch_pulsetype_label",
                            "seed_CPA_fluence", "seed_CPA_fluence_label", "seed_CPA_wavelength", "seed_CPA_wavelength_label", "seed_CPA_duration", "seed_CPA_duration_label", "seed_CPA_pulsetype", "seed_CPA_pulsetype_label",
                            "seed_type_button", "plot_seed_button"]
        self.toggle_sidebar_window(self.seed_button, self.seed_widgets)

        self.seed_type_button.set("Q-Switch")
        self.toggle_seed_type("Q-Switch")

    def load_amplifier_sidebar(self):
        row=30
        self.amplifier_title = App.create_label(self.settings_frame, text="Amplifier Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5), sticky=None)

        self.amplifier_passes, self.amplifier_passes_label = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="passes", init_val=Amplifier().passes, textwidget=True)
        self.amplifier_losses, self.amplifier_losses_label = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="losses [%]", init_val=Amplifier().losses*1e2, textwidget=True)
        self.amplifier_maxfluence, self.amplifier_maxfluence_label = App.create_entry(self.settings_frame, column=1, row=row+3, columnspan=2, width=110, text="max fluence [J/cm²]", init_val=Amplifier().max_fluence*1e-4, textwidget=True)

        self.amplifier_widgets = ["amplifier_title", "amplifier_passes", "amplifier_passes_label", "amplifier_losses", "amplifier_losses_label", "amplifier_maxfluence", "amplifier_maxfluence_label"]

        self.toggle_sidebar_window(self.amplifier_button, self.amplifier_widgets)

    # load the classes
    def load_crystal(self):
        material = self.material_list.get()
        temperature = int(self.temperature_list.get())
        N_dop = float(self.crystal_doping.get())*1e6
        length = float(self.crystal_thickness.get())*1e-3
        tau_f = float(self.crystal_tau_f.get())*1e-3
        return Crystal(material=material, temperature=temperature, N_dop=N_dop, length=length, tau_f=tau_f)
        
    def load_seed_pulse(self, type):
        if type == "Q-Switch":
            seed_pulse = Seed(fluence=float(self.seed_QSwitch_fluence.get()), wavelength=float(self.seed_QSwitch_wavelength.get()), duration=float(self.seed_QSwitch_duration.get()), seed_type=self.seed_QSwitch_pulsetype.get())
        elif type == "CPA":
            seed_pulse = Seed_CPA(fluence=float(self.seed_CPA_fluence.get()), wavelength=float(self.seed_CPA_wavelength.get()), bandwidth=float(self.seed_CPA_duration.get()), seed_type=self.seed_CPA_pulsetype.get())
        
        return seed_pulse

    def toggle_seed_type(self, value):
        if not self.seed_button.get():
            [getattr(self, name).grid_remove() for name in self.seed_widgets]
        elif value == "Q-Switch":
            [getattr(self, name).grid_remove() for name in self.seed_widgets if "CPA" in name]
            [getattr(self, name).grid() for name in self.seed_widgets if "QSwitch" in name]
        elif value == "CPA":
            [getattr(self, name).grid_remove() for name in self.seed_widgets if "QSwitch" in name]
            [getattr(self, name).grid() for name in self.seed_widgets if "CPA" in name]

    def toggle_sidebar_window(self, button, widgets):
        if button.get():
            self.settings_frame.grid()
            [getattr(self, name).grid() for name in widgets]
        else:
            [getattr(self, name).grid_remove() for name in widgets]
            self.close_sidebar_window()

        if button == self.seed_button:
            self.toggle_seed_type(self.seed_type_button.get())

    def close_sidebar_window(self):
        if not self.pump_button.get() and not self.seed_button.get() and not self.amplifier_button.get():
            self.settings_frame.grid_remove()

    # Plotting section
    def setup_plot_area(self):
        """Call this ONCE when initializing the GUI"""
        self.fig = plt.figure(constrained_layout=True, dpi=150)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
        self.canvas_widget = self.canvas.get_tk_widget()
        self.toolbar = self.create_toolbar()
        self.canvas_widget.pack(fill="both", expand=True)
        self.ax = self.fig.add_subplot(1, 1, 1)  # create axis once
        self.canvas.draw()
        print("Plot area set up")

    def clear_axis(self):
        self.ax.clear()
        for other_ax in self.fig.axes[:]:
            if other_ax is not self.ax:
                self.fig.delaxes(other_ax)

    def seed_plot(self):
        self.clear_axis()

        if self.seed_type_button.get() == "Q-Switch":
            plot_QSwitch_pulse(self.load_seed_pulse("Q-Switch"), axis=self.ax)
        else:
            plot_CPA_pulse(self.load_seed_pulse("CPA"), axis=self.ax)
        
        self.canvas.draw()

    def amplifier_plot(self):
        self.clear_axis()

        plot_function = self.amplifier_plot_functions[self.amplifier_plot_list.get()]

        if plot_function == plot_temporal_fluence:
            seed = self.load_seed_pulse("Q-Switch")
        elif plot_function == plot_spectral_fluence:
            seed = self.load_seed_pulse("CPA")
        elif plot_function == plot_inversion_before_after or plot_function == plot_total_fluence_per_pass:
            seed = self.load_seed_pulse(self.seed_type_button.get())
        else:
            seed = self.load_seed_pulse("Q-Switch")

        crystal = self.load_crystal()

        pump_res = round(numres*max(1,np.sqrt((float(self.pump_duration.get())*1e-3/crystal.tau_f))))

        pump = Pump(intensity=float(self.pump_intensity.get()), wavelength=float(self.pump_wavelength.get()), duration=float(self.pump_duration.get()), resolution=pump_res)

        amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed, passes=int(self.amplifier_passes.get()), losses=float(self.amplifier_losses.get())*1e-2, max_fluence=float(self.amplifier_maxfluence.get()))
        plot_function(amplifier, axis=[self.ax, self.fig], save_data=self.save_data_switch.get(), save_path=self.folder_path.get(), save=self.save_plot_switch.get())

        self.canvas.draw()

    def crystal_plot(self):
        self.clear_axis()

        crystal = self.load_crystal()
        lambda_p = float(self.pump_wavelength.get())
        lambda_l = float(self.seed_QSwitch_wavelength.get())

        plot_function = self.material_plot_functions[self.material_plot_list.get()]

        kwargs = {"axis": [self.ax, self.fig],
                  "save_data": self.save_data_switch.get(),
                  "save_path": self.folder_path.get(),
                  "save": self.save_plot_switch.get()}  # base args for all plots

        if plot_function == plot_small_signal_gain:
            kwargs["beta"] = ast.literal_eval(self.inversion.get())
        elif plot_function == plot_Isat:
            kwargs.update({"lambda0": lambda_p, "xlim": (lambda_p - 30, lambda_p + 30)})
        elif plot_function == plot_Fsat:
            kwargs.update({"lambda0": lambda_l, "xlim": (lambda_l - 30, lambda_l + 30)})
        elif plot_function == plot_cross_sections and self.plot_pump_laser_cross_sections.get():
            kwargs.update({"lambda_p": lambda_p, "lambda_l": lambda_l})

        # one single call
        plot_function(crystal, **kwargs)
        self.canvas.draw()

    def update_material(self, material):
        material_path = os.path.join(LaserSim_path, "material_database", material)
        file_name = [f for f in os.listdir(material_path)]
        temperatures = []
        for s in file_name:
            matches = re.findall(r"\d+", s)
            if matches:
                temperatures.append(matches[-1])  # take the last match

        # Step 3: Remove duplicates (convert to set, then back to list if needed)
        temperatures = list(sorted(set(temperatures)))
        print(temperatures)
        self.temperature_list.configure(values=temperatures)
        if self.temperature_list.get() not in temperatures:
            self.temperature_list.set(temperatures[0])

        self.crystal_doping.reinsert(str(Crystal(material=material).doping_concentration*1e-6))
        self.crystal_thickness.reinsert(str(Crystal(material=material).length*1e3))
        self.crystal_tau_f.reinsert(str(Crystal(material=material).tau_f*1e3))

    def toggle_extra_material_arguments(self, argument):
        if argument == "Small signal gain":
            self.inversion.grid()
            self.inversion_label.grid()
        else:
            self.inversion.grid_remove()
            self.inversion_label.grid_remove()

        if argument == "Cross sections":
            self.plot_pump_laser_cross_sections.grid()
        else:
            self.plot_pump_laser_cross_sections.grid_remove()

    def read_file_list(self):
        path = customtkinter.filedialog.askdirectory(initialdir=self.folder_path)
        if path != "":
            self.folder_path.reinsert(path)

    def create_toolbar(self) -> customtkinter.CTkFrame:
        # toolbar_frame = customtkinter.CTkFrame(master=self.tabview.tab("Show Plots"))
        # toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar_frame = customtkinter.CTkFrame(self)
        toolbar_frame.grid(row=20, column=1, columnspan=2, padx=(10), sticky="ew")
        toolbar = CustomToolbar(self.canvas, toolbar_frame)
        toolbar.config(background=self.color)
        toolbar._message_label.config(background=self.color, foreground=self.text_color, font=(15))
        toolbar.winfo_children()[-2].config(background=self.color)
        toolbar.update()
        return toolbar_frame

        # Closing the application       
    
    def toggle_toolbar(self, value):
        if value == "Show Plots":
            self.toolbar.grid()
        else:
            self.toolbar.grid_remove()
    # Save the current figure or the data based on the file type
    def save_figure(self):
        file_name = customtkinter.filedialog.asksaveasfilename()
        self.fig.savefig(file_name, bbox_inches='tight')

    def save_project(self, filename):
        # collect all data-variables to be saved into a dictionary
        project_data = {}
        for name in self.save_attributes:
            val = getattr(self, name)
            # unwrap Tkinter variables automatically
            if isinstance(val, (
                customtkinter.CTkLabel,
                customtkinter.CTkButton,
                customtkinter.CTkSwitch,
                customtkinter.CTkSegmentedButton,
                customtkinter.CTkFrame,
                customtkinter.CTkTextbox
            )):
                continue
            if hasattr(val, "get"):
                val = val.get()
            project_data[name] = val
        
        print(project_data)

        # Read out the variables and convert to JSON-safe dict
        def to_json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()      # convert numpy arrays to lists
            if isinstance(obj, (np.int64, np.float64)):
                return obj.item()        # convert numpy scalars to Python
            if isinstance(obj, dict):
                return {k: to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_json_safe(v) for v in obj]
            return obj                   # base case

        safe_data = to_json_safe(project_data)

        with open(filename, "w") as f:
            json.dump(safe_data, f, indent=2)

    def load_project(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        print(data)
        for name, val in data.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if hasattr(attr, "set"):  # Tkinter variable
                    attr.set(val)
                elif hasattr(attr, "reinsert"):
                    attr.reinsert(val)

    def on_closing(self):
        self.quit()    # Python 3.12 works
        self.destroy() # needed for built exe

class CustomToolbar(NavigationToolbar2Tk):
    # Modify the toolitems list to remove specific buttons
    def set_message(self, s):
        formatted_message = s.replace("\n", ", ").strip()
        self.message.set(formatted_message)

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # Add or remove toolitems as needed
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        # ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )

# Extend the class of the entry widget to add the reinsert method
class CustomEntry(customtkinter.CTkEntry):
    def reinsert(self, text):
        self.delete(0, 'end')  # Delete the current text
        self.insert(0, text)  # Insert the new text

if __name__ == "__main__":

    app = App()
    app.state('zoomed')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()