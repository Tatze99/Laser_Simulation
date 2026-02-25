import os
import re
import sys
import json
from CTkRangeSlider import *
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image
import ast # for literal_eval of a user input

from LaserSim.crystal import Crystal, plot_cross_sections, plot_small_signal_gain, plot_beta_eq, plot_Isat, plot_Fsat, plot_lambert_beer
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.spectral_losses import Spectral_Losses
from LaserSim.amplifier import Amplifier, plot_inversion1D, plot_inversion_temporal, plot_inversion2D, plot_temporal_fluence, plot_total_fluence_per_pass, plot_spectral_fluence, plot_inversion_before_after, plot_inversion_vs_pump_intensity, plot_storage_efficiency_vs_pump_time, plot_storage_efficiency_2D, plot_storage_efficiency_vs_pump_intensity, plot_pump_absorption
from LaserSim.utilities import set_plot_params, numres, LaserSimFolder
from LaserSim.seed import plot_seed_pulse as plot_QSwitch_pulse
from LaserSim.seed_CPA import plot_seed_pulse as plot_CPA_pulse

from LaserSim.spectral_losses import test_reflectivity_approximation


version_number = "26/02"
Standard_path = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(LaserSimFolder, "material_database")

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
        self.toggle_extra_material_arguments("Cross sections")

        self.canvas_width.bind("<KeyRelease>", lambda val: self.update_canvas_size(self.canvas_ratio_list[self.canvas_ratio.get()]))
        self.canvas_height.bind("<KeyRelease>", lambda val: self.update_canvas_size(self.canvas_ratio_list[self.canvas_ratio.get()]))

        self.save_attributes = ["material_list", "temperature_list", "material_plot_list", "amplifier_plot_list"]

        self.save_attributes.extend(["crystal_doping", "crystal_thickness", "crystal_tau_f", "crystal_ZPL"])
        self.save_attributes.extend(["pump_intensity", "pump_wavelength", "pump_duration"])
        self.save_attributes.extend(["seed_type_button", "seed_QSwitch_fluence", "seed_QSwitch_wavelength", "seed_QSwitch_duration", "seed_QSwitch_pulsetype", "seed_CPA_fluence", "seed_CPA_wavelength", "seed_CPA_bandwidth", "seed_CPA_pulsetype", "seed_gaussian_order"])
        self.save_attributes.extend(["amplifier_passes", "amplifier_losses", "amplifier_maxfluence"])
        self.save_attributes.extend(self.settings_widgets)

    def initialize_ui_images(self):
        self.img_settings = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","options.png")), size=(15, 15))
        self.img_save = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","save_white.png")), size=(15, 15))
        self.img_folder = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","folder.png")), size=(15, 15))
        self.img_reset = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","reset.png")), size=(15, 15))
        self.img_crystal = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","crystal.png")), size=(15, 15))
        self.img_laser = customtkinter.CTkImage(dark_image=Image.open(os.path.join(Standard_path,"ui_images","laser.png")), size=(15, 15))

    def initialize_variables(self):
        self.materials = [f for f in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, f)) and not f.startswith('plots') and not f.startswith('reflectivity_curves')]
        self.material_plot_functions = {'Cross sections': plot_cross_sections, 
                                        'Small signal gain': plot_small_signal_gain,
                                        'Equilibrium inversion': plot_beta_eq, 
                                        'Saturation intensity': plot_Isat, 
                                        'Saturation fluence': plot_Fsat,
                                        'Lambert-Beer transmission': plot_lambert_beer
                                        }
        
        self.amplifier_plot_functions = {'Temporal fluence': plot_temporal_fluence, 
                                        'Total fluence pass': plot_total_fluence_per_pass,
                                        'Spectral fluence': plot_spectral_fluence,
                                        'Inversion 1D (space)': plot_inversion1D, 
                                        'Inversion 1D (time)': plot_inversion_temporal,
                                        'Inversion 2D': plot_inversion2D,
                                        'Inversion vs Ip': plot_inversion_vs_pump_intensity, 
                                        'Inversion before vs after': plot_inversion_before_after,
                                        'Pump absorption': plot_pump_absorption,
                                        'Storage efficiency vs t': plot_storage_efficiency_vs_pump_time,
                                        'Storage efficiency vs Ip': plot_storage_efficiency_vs_pump_intensity,
                                        'Storage efficiency 2D': plot_storage_efficiency_2D
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
        self.canvas_ratio_list = {'Auto': None, 'Custom': 0,'4:3 ratio': 4/3, '16:9 ratio': 16/9, '3:2 ratio': 3/2, '3:1 ratio': 3,'2:1 ratio': 2, '1:1 ratio': 1, '1:2 ratio': 0.5}

    # user interface, gets called when the program starts 
    def initialize_ui(self):

        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, height=600, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nesw")
        self.sidebar_frame.rowconfigure(20, weight=1)
        self.rowconfigure(11,weight=1)

        App.create_label(self.sidebar_frame, text="LaserSim v."+version_number, font=customtkinter.CTkFont(size=20, weight="bold"),row=0, column=0, padx=20, pady=(10,15), sticky=None)
        
        self.tabview = customtkinter.CTkTabview(self, width=250, command=lambda: self.toggle_toolbar(self.tabview.get()))
        self.tabview.grid(row=3, column=1, padx=(10, 10), pady=(20, 0), columnspan=2, sticky="nsew", rowspan=10)
        self.tabview.add("Show Plots")
        self.tabview.add("Settings")
        self.tabview.tab("Show Plots").columnconfigure(0, weight=1)
        self.tabview.tab("Show Plots").rowconfigure(0, weight=1)

        #buttons
        
        frame = self.sidebar_frame

        self.material_list  = App.create_Menu(frame, values=self.materials, column=0, row=1, command=self.update_material, width=135, padx = (10,210-135), init_val="YbCaF2")
        self.temperature_list  = App.create_Menu(frame, values=["300"], column=0, row=1, command=None, width=60, padx = (210-60,10))

        self.material_plot_list  = App.create_Menu(frame, values=list(self.material_plot_functions.keys()), column=0, row=2, pady=(15,5), command=self.toggle_extra_material_arguments)
        self.plot_crystal_button    = App.create_button(frame, text="Plot material", command=self.crystal_plot, column=0, row=4, image=self.img_crystal, sticky="w")
        
        self.amplifier_plot_list  = App.create_Menu(frame, values=list(self.amplifier_plot_functions.keys()), column=0, row=7, pady=(25,5), command=self.toggle_extra_amplifier_arguments)
        self.plot_amplifier_button = App.create_button(frame, text="Plot amplifier", command=self.amplifier_plot, column=0, row=8, image=self.img_laser, sticky="w")
        
        
        # extra settings
        self.inversion    = App.create_entry(frame, column=0, row=5, init_val="0.1,0.15,0.2", width=170, padx=(210-170, 10))
        self.double_pass = App.create_switch(frame, text="Double pass", command=None,  column=0, row=6, padx=20)
        self.inversion_label    = App.create_label(frame, column=0, row=5, text="β", padx=(5, 215-20))
        self.plot_pump_laser_cross_sections = App.create_switch(frame, text="Show values at λp, λl", command=None,  column=0, row=5, padx=20)
        self.add_legend   = App.create_entry(frame, column=0, row=9, init_val="", width=150, padx=(210-150, 10))
        self.add_legend_label    = App.create_label(frame, column=0, row=9, text="legend", padx=(10, 210-40))
        self.reset_material_plot = App.create_button(frame, width=50, command=lambda: (self.clear_figure(), self.crystal_plot()), column=0, row=4, image=self.img_reset, sticky="e")
        self.reset_amplifier_plot = App.create_button(frame, width=50, command=lambda: (self.clear_figure(), self.amplifier_plot()), column=0, row=8, image=self.img_reset, sticky="e")
        self.normalize = App.create_switch(frame, text="Normalize", command=None,  column=0, row=9, padx=20, pady=(5,15))

        # bottom settings
        self.save_button    = App.create_button(frame, text="Save figure/data", command=self.save_figure,     column=0, row=23,  image=self.img_save, pady=(5,15))
        
        #switches
        self.config_title = App.create_label(frame, text="Configure Simulation", font=customtkinter.CTkFont(size=16, weight="bold"), row=10, column=0, padx=20, pady=(20, 5),sticky=None)
        self.crystal_button   = App.create_switch(frame, text="Config Crystal", command=lambda: self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets),  column=0, row=11, padx=20)
        self.pump_button      = App.create_switch(frame, text="Config Pump", command=lambda: self.toggle_sidebar_window(self.pump_button, self.pump_widgets),  column=0, row=12, padx=20)
        self.seed_button      = App.create_switch(frame, text="Config Seed", command=lambda: self.toggle_sidebar_window(self.seed_button, self.seed_widgets),  column=0, row=13, padx=20)
        self.amplifier_button = App.create_switch(frame, text="Config Amplifier", command=lambda: self.toggle_sidebar_window(self.amplifier_button, self.amplifier_widgets),  column=0, row=14, padx=20)
        self.multiplot_button   = App.create_switch(frame, text="Multiplot", command= self.toggle_multiplot_buttons,  column=0, row=15, padx=20, pady=(5,15))

        #Settings section
        frame = self.tabview.tab("Settings")
        self.load_button = App.create_button(frame, text="Set Working directory", command=self.read_file_list,  column=0, row=0, columnspan=2, image=self.img_folder, width=250)
        self.folder_path = App.create_entry(frame, row=0, column=3, text="Folder path", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        self.json_path   = App.create_entry(frame, row=1, column=3, text="project file name", columnspan=2, width=600, padx=10, pady=10, sticky="w")
        self.folder_path.insert(0, Standard_path)
        self.json_path.insert(0, "project_data")
        self.save_data_button    = App.create_button(frame, text="save project", command=lambda: self.save_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")),  column=0, row=1, image=self.img_save, width=110)
        self.load_data_button    = App.create_button(frame, text="load project", command=lambda: self.load_project(os.path.join(self.folder_path.get(), f"{self.json_path.get()}.json")),  column=1, row=1, image=self.img_folder, width=110)

        self.plot_settings_title = App.create_label(frame, text="Plot settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=2, column=0, columnspan=2, padx=20, pady=(20, 5),sticky=None)
        self.show_title = App.create_switch(frame, text="Show title", command=None,  column=0, row=3, padx=20, pady=(10,5), columnspan=2)
        self.show_grid = App.create_switch(frame, text="Use Grid", command=self.toggle_grid,  column=0, row=4, padx=20, columnspan=2)
        self.save_data = App.create_switch(frame, text="Save data during plot", command=None,  column=0, row=5, padx=20, columnspan=2)
        self.save_plot = App.create_switch(frame, text="Save figure during plot", command=None,  column=0, row=6, padx=20, columnspan=2)

        self.cross_section_settings_title = App.create_label(frame, text="Cross section settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=2, column=2, columnspan=2, padx=20, pady=(20, 5),sticky=None)
        self.smooth_sigma = App.create_switch(frame, text="Smooth cross sections", command=None,  column=2, row=3, padx=20, pady=(10,5), columnspan=2)
        self.McCumber_absorption = App.create_switch(frame, text="Use McCumber absorption", command=None,  column=2, row=4, padx=20, pady=(10,5), columnspan=2)

        self.canvas_size_title = App.create_label(frame, text="Canvas Size", font=customtkinter.CTkFont(size=16, weight="bold"), row=9, column=0, columnspan=2, padx=20, pady=(20, 5),sticky=None)
        self.canvas_width, self.canvas_width_label        = App.create_entry(frame,column=1, row=11, width=70,text="width in cm", placeholder_text="10 [cm]", sticky='w', init_val=10, textwidget=True)
        self.canvas_height, self.canvas_height_label      = App.create_entry(frame,column=1, row=12, width=70,text="height in cm", placeholder_text="10 [cm]", sticky='w', init_val=10, textwidget=True)
        self.canvas_ratio   = App.create_Menu(frame, column=1, row=10, width=110, values=list(self.canvas_ratio_list.keys()), text="Canvas Size", command=lambda x: self.update_canvas_size(self.canvas_ratio_list[x]))

        self.settings_widgets = ["save_data", "save_plot", "crystal_button", "pump_button", "seed_button", "amplifier_button", "show_title", "show_grid", "canvas_width", "canvas_height", "canvas_ratio"]

        self.normalize.grid_remove()
        self.inversion.grid_remove()
        self.double_pass.grid_remove()
        self.inversion_label.grid_remove()
        self.plot_pump_laser_cross_sections.grid_remove()
        self.add_legend.grid_remove()
        self.add_legend_label.grid_remove()
        self.reset_material_plot.grid_remove()
        self.reset_amplifier_plot.grid_remove()
        self.show_title.select()
        self.double_pass.select()
        self.show_grid.select()
        self.smooth_sigma.select()
        self.McCumber_absorption.select()

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

    def create_switch(self, command, row, column, text, columnspan=1, padx=10, pady=5, sticky='w',init_on = False, **kwargs):
        switch = customtkinter.CTkSwitch(self, text=text, command=command, **kwargs)
        switch.grid(row=row, column=column, columnspan=columnspan, padx=padx, pady=pady, sticky=sticky, **kwargs)
        if init_on: switch.select()
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
        before = set(self.settings_frame.winfo_children())
        self.crystal_title = App.create_label(self.settings_frame, text="Crystal Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.crystal_doping = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="doping [cm⁻³]")
        self.crystal_thickness = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="thickness [mm]")
        self.crystal_tau_f = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="lifetime τ [ms]")
        self.crystal_ZPL = App.create_entry(self.settings_frame,column=1, row=row+4, columnspan=2, width=110, text="ZPL [nm]")

        self.crystal_widgets = set(self.settings_frame.winfo_children()) - before
        self.toggle_sidebar_window(self.crystal_button, self.crystal_widgets)

    def load_pump_sidebar(self):
        row = 10
        before = set(self.settings_frame.winfo_children())
        self.pump_title = App.create_label(self.settings_frame, text="Pump Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.pump_wavelength = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="wavelength [nm]", init_val=round(Pump().wavelength*1e9, 3))
        
        self.pump_intensity = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110-50, text="intensity [kW/cm²]", init_val=Pump().intensity*1e-7, padx=(10,10+50))
        self.pump_intensity_button = App.create_button(self.settings_frame,column=1, row=row+2, columnspan=2, width=40, text="Isat", command=lambda: self.pump_intensity.reinsert(round(Crystal(material=self.material).I_sat(float(self.pump_wavelength.get())*1e-9)*1e-7,2)), padx=(80,10))

        self.pump_duration = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110-50, text="duration [ms]", init_val=Pump().duration*1e3, padx=(10,10+50))
        self.pump_duration_button = App.create_button(self.settings_frame,column=1, row=row+3, columnspan=2, width=40, text="τ", command=lambda: self.pump_duration.reinsert(self.crystal_tau_f.get()), padx=(80,10))

        self.pump_widgets= set(self.settings_frame.winfo_children()) - before
        self.toggle_sidebar_window(self.pump_button, self.pump_widgets)

    def load_seed_sidebar(self):
        row = 20
        before = set(self.settings_frame.winfo_children())
        self.seed_title = App.create_label(self.settings_frame, text="Seed Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5),sticky=None)
        self.seed_type_button = App.create_segmented_button(self.settings_frame, values=["Q-Switch","CPA"], command=lambda value: self.toggle_seed_type(value), row=row+1, column=1, columnspan=3, width=110)
        self.seed_type_label = App.create_label(self.settings_frame, text="Seed Type", column=0, row=row+1)

        Q_switch_start = set(self.settings_frame.winfo_children())
        self.seed_QSwitch_fluence     = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="fluence [J/cm²]", init_val=Seed().fluence*1e-4)
        self.seed_QSwitch_wavelength  = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="wavelength [nm]", init_val=Seed().wavelength*1e9)
        self.seed_QSwitch_duration    = App.create_entry(self.settings_frame,column=1, row=row+4, columnspan=2, width=110, text="duration [ns]", init_val=Seed().duration*1e9)
        self.seed_QSwitch_pulsetype   = App.create_Menu(self.settings_frame, values=["gauss","lorentz","rect"], column=1, row=row+5, command=self.toggle_extra_seed_arguments, text="pulse type", width=110)
        self.seed_gaussian_order, self.seed_gaussian_order_label = App.create_entry(self.settings_frame,column=1, row=row+6, columnspan=2, width=110, text="gaussian order", init_val=1, textwidget=True) # need the label!
        Q_switch_end = set(self.settings_frame.winfo_children())

        self.seed_CPA_fluence         = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="fluence [J/cm²]", init_val=Seed_CPA().fluence*1e-4)
        self.seed_CPA_wavelength      = App.create_entry(self.settings_frame,column=1, row=row+3, columnspan=2, width=110, text="wavelength [nm]", init_val=round(Seed_CPA().wavelength*1e9, 3))
        self.seed_CPA_bandwidth        = App.create_entry(self.settings_frame,column=1, row=row+4, columnspan=2, width=110, text="bandwidth [nm]", init_val=round(Seed_CPA().bandwidth*1e9, 3))
        self.seed_CPA_pulsetype       = App.create_Menu(self.settings_frame, values=["gauss","lorentz","rect"], column=1, row=row+5, command=self.toggle_extra_seed_arguments, text="pulse type", width=110)
        CPA_end = set(self.settings_frame.winfo_children())

        self.plot_seed_button = App.create_button(self.settings_frame, text="Plot Seed Pulse", command=self.seed_plot, row=row+7, column=0, columnspan=5, padx=20, pady=(5, 15))

        self.seed_widgets= set(self.settings_frame.winfo_children()) - before
        self.Q_Switch_widgets = Q_switch_end - Q_switch_start
        self.CPA_widgets = CPA_end - Q_switch_end
        self.toggle_sidebar_window(self.seed_button, self.seed_widgets)

        self.seed_type_button.set("Q-Switch")
        self.toggle_seed_type("Q-Switch")

    def load_amplifier_sidebar(self):
        row=30
        before = set(self.settings_frame.winfo_children())
        self.amplifier_title = App.create_label(self.settings_frame, text="Amplifier Settings", font=customtkinter.CTkFont(size=16, weight="bold"), row=row, column=0, columnspan=5, padx=20, pady=(20, 5), sticky=None)

        self.amplifier_passes = App.create_entry(self.settings_frame,column=1, row=row+1, columnspan=2, width=110, text="passes", init_val=Amplifier().passes)
        self.amplifier_losses = App.create_entry(self.settings_frame,column=1, row=row+2, columnspan=2, width=110, text="losses per RT [%]", init_val=Amplifier().losses*1e2)
        self.amplifier_maxfluence = App.create_entry(self.settings_frame, column=1, row=row+3, columnspan=2, width=110, text="max fluence [J/cm²]", init_val=Amplifier().max_fluence*1e-4)
        self.amplifier_double_pass = App.create_switch(self.settings_frame, command=None, column=1, row=row+4, columnspan=2, text="double pass", init_on=True)

        self.amplifier_widgets = set(self.settings_frame.winfo_children()) - before
        self.toggle_sidebar_window(self.amplifier_button, self.amplifier_widgets)

    # load the classes
    def load_crystal(self):
        material = self.material_list.get()
        temperature = int(self.temperature_list.get())
        N_dop = float(self.crystal_doping.get())*1e6
        length = float(self.crystal_thickness.get())*1e-3
        tau_f = float(self.crystal_tau_f.get())*1e-3

        crystal = Crystal(material=material, temperature=temperature, N_dop=N_dop, length=length, tau_f=tau_f, smooth_sigmas=self.smooth_sigma.get(), useMcCumber_absorption=self.McCumber_absorption.get())
        crystal.lambda_ZPL = float(self.crystal_ZPL.get())*1e-9
        return crystal

    def load_pump(self, crystal=None):
        if crystal: pump_res = round(numres*max(1,np.sqrt((float(self.pump_duration.get())*1e-3/crystal.tau_f))))
        intensity = float(self.pump_intensity.get())
        wavelength = float(self.pump_wavelength.get())
        duration = float(self.pump_duration.get())

        return Pump(intensity=intensity, wavelength=wavelength, duration=duration, resolution=pump_res)
      
    def load_seed_pulse(self, type):
        if type == "Q-Switch":
            seed_pulse = Seed(fluence=float(self.seed_QSwitch_fluence.get()), wavelength=float(self.seed_QSwitch_wavelength.get()), duration=float(self.seed_QSwitch_duration.get()), seed_type=self.seed_QSwitch_pulsetype.get(), gauss_order=int(self.seed_gaussian_order.get()))
        elif type == "CPA":
            seed_pulse = Seed_CPA(fluence=float(self.seed_CPA_fluence.get()), wavelength=float(self.seed_CPA_wavelength.get()), bandwidth=float(self.seed_CPA_bandwidth.get()), seed_type=self.seed_CPA_pulsetype.get(), gauss_order=int(self.seed_gaussian_order.get()))
        
        return seed_pulse

    def toggle_seed_type(self, value):
        if not self.seed_button.get():
            [widget.grid_remove() for widget in self.seed_widgets]
        elif value == "Q-Switch":
            [widget.grid_remove() for widget in self.CPA_widgets]
            [widget.grid() for widget in self.Q_Switch_widgets]
            self.toggle_extra_seed_arguments(self.seed_QSwitch_pulsetype.get())
        elif value == "CPA":
            [widget.grid_remove() for widget in self.Q_Switch_widgets]
            [widget.grid() for widget in self.CPA_widgets]
            self.toggle_extra_seed_arguments(self.seed_CPA_pulsetype.get())

    def toggle_sidebar_window(self, button, widgets):
        if button.get():
            self.settings_frame.grid()
            [widget.grid() for widget in widgets]
        else:
            [widget.grid_remove() for widget in widgets]
            self.close_sidebar_window()

        if button == self.seed_button:
            self.toggle_seed_type(self.seed_type_button.get())

    def toggle_multiplot_buttons(self):
        if self.multiplot_button.get():
            self.plot_crystal_button.configure(width=140)
            self.plot_amplifier_button.configure(width=140)
            self.reset_amplifier_plot.grid()
            self.reset_material_plot.grid()
        else:
            self.plot_crystal_button.configure(width=200)
            self.plot_amplifier_button.configure(width=200)
            self.reset_amplifier_plot.grid_remove()
            self.reset_material_plot.grid_remove()

    def toggle_grid(self):
        self.ax.grid(self.show_grid.get())
        plt.rcParams["axes.grid"] = self.show_grid.get()

    def close_sidebar_window(self):
        if not self.crystal_button.get() and not self.pump_button.get() and not self.seed_button.get() and not self.amplifier_button.get():
            self.settings_frame.grid_remove()

    # Plotting section
    def setup_plot_area(self):
        """Call this ONCE when initializing the GUI"""
        self.fig = plt.figure(constrained_layout=True, dpi=150)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tabview.tab("Show Plots"))
        self.canvas_widget = self.canvas.get_tk_widget()
        self.toolbar = self.create_toolbar()
        self.canvas_widget.pack(fill="both", expand=True)
        self.canvas.draw()

    def clear_figure(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def clear_axis(self):
        if not self.multiplot_button.get():
            self.clear_figure()
        else:
            self.ax.relim()
            self.ax.autoscale()

        self.kwargs = {"axis": [self.ax, self.fig],
                       "save_data": self.save_data.get(),
                       "save_path": self.folder_path.get(),
                       "save": self.save_plot.get(),
                       "show_title": self.show_title.get(),
                    #    "kwargs": {"linestyle": '-', "color": "tab:blue"}
                    }  # base args for all plots

    def seed_plot(self):
        self.clear_axis()

        if self.seed_type_button.get() == "Q-Switch":
            plot_QSwitch_pulse(self.load_seed_pulse("Q-Switch"), **self.kwargs)
        else:
            plot_CPA_pulse(self.load_seed_pulse("CPA"), **self.kwargs)
        
        self.canvas.draw()

    def amplifier_plot(self):
        self.clear_axis()

        plot_function = self.amplifier_plot_functions[self.amplifier_plot_list.get()]
        seed_type = "Q-Switch"
        kwargs = dict()

        if plot_function == plot_temporal_fluence:
            seed_type = "Q-Switch"
            kwargs["normalize"] = self.normalize.get()
        elif plot_function == plot_spectral_fluence:
            seed_type = "CPA"
            kwargs["normalize"] = self.normalize.get()
        elif plot_function == plot_inversion_before_after or plot_function == plot_total_fluence_per_pass:
            seed_type = self.seed_type_button.get()
        
        if plot_function in [plot_total_fluence_per_pass, plot_inversion1D, plot_inversion_temporal, plot_inversion2D, plot_inversion_vs_pump_intensity, plot_storage_efficiency_2D, plot_storage_efficiency_vs_pump_intensity]:
            kwargs["custom_legend"] = self.add_legend.get().format(crystal = self.material_list.get(),
                                                                   intensity=float(self.pump_intensity.get()), 
                                                                   tau_p=float(self.pump_duration.get()),
                                                                   Ndop=float(self.crystal_doping.get()),
                                                                   thickness=float(self.crystal_thickness.get()),
                                                                   temperature=self.temperature_list.get(),
                                                                   ZPL=float(self.crystal_ZPL.get()),
                                                                   tau_f=float(self.crystal_tau_f.get()),
                                                                   losses=float(self.amplifier_losses.get()),
                                                                   lambda_p=float(self.pump_wavelength.get()),
                                                                   lambda_l=float(self.seed_QSwitch_wavelength.get()))
        if plot_function == plot_storage_efficiency_vs_pump_time:
            kwargs["pump_intensity"] = [float(self.pump_intensity.get())*1e7]

        seed = self.load_seed_pulse(seed_type)
        crystal = self.load_crystal()
        pump = self.load_pump(crystal=crystal)

        amplifier = Amplifier(crystal=crystal, pump=pump, seed=seed, passes=int(self.amplifier_passes.get()), losses=float(self.amplifier_losses.get())*1e-2, max_fluence=float(self.amplifier_maxfluence.get()), double_pass=self.amplifier_double_pass.get())
        plot_function(amplifier, **self.kwargs, **kwargs)

        self.canvas.draw()

    def crystal_plot(self):
        self.clear_axis()

        crystal = self.load_crystal()
        lambda_p = crystal.to_display_lambda(float(self.pump_wavelength.get())*1e-9)
        lambda_l = crystal.to_display_lambda(float(self.seed_QSwitch_wavelength.get())*1e-9)

        plot_function = self.material_plot_functions[self.material_plot_list.get()]

        kwargs = dict()
        bandwidth = crystal.to_display_lambda(float(self.seed_CPA_bandwidth.get())*1e-9)

        if plot_function == plot_small_signal_gain:
            kwargs["beta"] = ast.literal_eval(self.inversion.get())
            kwargs["double_pass"] = self.double_pass.get()
            kwargs["xlim"] = (lambda_l - bandwidth, lambda_l + bandwidth)
        elif plot_function == plot_Isat:
            kwargs.update({"lambda0": lambda_p, "xlim": (lambda_p - bandwidth, lambda_p + bandwidth)})
        elif plot_function == plot_Fsat:
            kwargs.update({"lambda0": lambda_l, "xlim": (lambda_l - bandwidth, lambda_l + bandwidth)})
        elif (plot_function == plot_cross_sections or plot_function == plot_beta_eq) and self.plot_pump_laser_cross_sections.get():
            kwargs.update({"lambda_p": lambda_p, "lambda_l": lambda_l})
        elif plot_function == plot_lambert_beer:
            kwargs.update({"lambda0": lambda_p})

        # one single call
        plot_function(crystal, **self.kwargs, **kwargs)
        self.canvas.draw()

    def update_material(self, material):
        material_path = os.path.join(database_path, material)
        file_name = [f for f in os.listdir(material_path)]
        temperatures = []
        for s in file_name:
            matches = re.findall(r"\d+", s)
            if matches:
                temperatures.append(matches[-1])  # take the last match

        # Step 3: Remove duplicates (convert to set, then back to list if needed)
        temperatures = list(sorted(set(temperatures)))
        self.temperature_list.configure(values=temperatures)
        if self.temperature_list.get() not in temperatures:
            self.temperature_list.set(temperatures[0])

        self.material = material
        crystal = Crystal(material=material, smooth_sigmas=self.smooth_sigma.get())
        self.crystal_doping.reinsert(str(crystal.doping_concentration*1e-6))
        self.crystal_thickness.reinsert(str(crystal.length*1e3))
        self.crystal_tau_f.reinsert(str(crystal.tau_f*1e3))
        self.crystal_ZPL.reinsert(str(crystal.lambda_ZPL*1e9))
        self.seed_QSwitch_wavelength.reinsert(f"{crystal.to_internal_lambda(crystal.lambda_e)*1e9:g}")
        self.seed_CPA_wavelength.reinsert(f"{crystal.to_internal_lambda(crystal.lambda_e)*1e9:g}")
        self.pump_wavelength.reinsert(f"{crystal.to_internal_lambda(crystal.lambda_a)*1e9:g}")

    def toggle_extra_material_arguments(self, argument):
        if argument == "Small signal gain":
            self.inversion.grid()
            self.inversion_label.grid()
            self.double_pass.grid()
        else:
            self.inversion.grid_remove()
            self.inversion_label.grid_remove()
            self.double_pass.grid_remove()

        if argument == "Cross sections" or argument == "Equilibrium inversion":
            self.plot_pump_laser_cross_sections.grid()
        else:
            self.plot_pump_laser_cross_sections.grid_remove()
    
    def toggle_extra_amplifier_arguments(self, argument):
        if argument in ["Total fluence pass", "Storage efficiency 2D", "Inversion 1D (space)", "Inversion 1D (time)", "Inversion 2D", "Inversion vs Ip", "Storage efficiency vs Ip"]:
            self.add_legend.grid()
            self.add_legend_label.grid()
        else:
            self.add_legend.grid_remove()
            self.add_legend.delete(0, 'end')
            self.add_legend_label.grid_remove()

        self.normalize.grid() if (argument == "Temporal fluence" or argument == "Spectral fluence") else self.normalize.grid_remove()
    
    def toggle_extra_seed_arguments(self, argument):
        if argument == "gauss":
            self.seed_gaussian_order.grid()
            self.seed_gaussian_order_label.grid()
        else:
            self.seed_gaussian_order.grid_remove()
            self.seed_gaussian_order_label.grid_remove()

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
        if file_name.endswith((".pdf",".png",".jpg",".jpeg",".PNG",".JPG",".svg")): 
            self.fig.savefig(file_name, bbox_inches='tight')
        elif file_name.endswith((".dat",".txt",".csv")):
            all_data = []
            headers = []

            # Collect data
            for i, ax in enumerate(self.fig.axes):
                for j, line in enumerate(ax.get_lines()):
                    x = line.get_xdata()
                    y = line.get_ydata()
                    # make sure lengths match if different lines differ
                    length = min(len(x), len(y))
                    all_data.append(np.column_stack([x[:length], y[:length]]))
                    headers.extend([f"X{i}_{j}", f"Y{i}_{j}"])

            # Align all datasets by rows (pad with blanks if needed)
            max_len = max(arr.shape[0] for arr in all_data)
            aligned = np.full((max_len, len(all_data)*2), "", dtype=object)

            for k, arr in enumerate(all_data):
                aligned[:arr.shape[0], 2*k:2*k+2] = arr

            # Write to file
            with open(file_name, "w") as f:
                f.write("\t".join(headers) + "\n")
                for row in aligned:
                    f.write("\t".join(f"{float(val):.5e}" if val != "" else "" for val in row) + "\n")

    def save_project(self, filename):
        # collect all data-variables to be saved into a dictionary
        project_data = {}
        for name in self.save_attributes:
            val = getattr(self, name)
            # unwrap Tkinter variables automatically
            if isinstance(val, (
                customtkinter.CTkLabel,
                customtkinter.CTkButton,
                customtkinter.CTkSegmentedButton,
                customtkinter.CTkFrame,
                customtkinter.CTkTextbox
            )):
                continue
            if hasattr(val, "get"):
                val = val.get()
            project_data[name] = val

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

        for name, val in data.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if hasattr(attr, "set"):  # Tkinter variable
                    attr.set(val)
                elif hasattr(attr, "reinsert"):
                    attr.reinsert(val)
                elif hasattr(attr, "select") and val == 1:
                    attr.select()
                elif hasattr(attr, "deselect") and val == 0:
                    attr.deselect()
        
        self.close_sidebar_window()

    def update_canvas_size(self, canvas_ratio):
        canvas_width = float(self.canvas_width.get())
        canvas_height = float(self.canvas_height.get())
        if canvas_width <= 2 or canvas_height <= 2: return
        self.canvas_widget.pack_forget()

        self.canvas_height.grid() if canvas_ratio == 0 else self.canvas_height.grid_remove()
        self.canvas_height_label.grid() if canvas_ratio == 0 else self.canvas_height_label.grid_remove()

        if canvas_ratio is not None:
            width = canvas_width/2.54
            height = canvas_height/2.54 if canvas_ratio == 0 else canvas_width/(canvas_ratio*2.54)
            self.fig.set_size_inches(width, height)
            self.canvas_widget.pack(expand=True, fill=None) 
            self.canvas_widget.config(width=self.fig.get_size_inches()[0] * self.fig.dpi, height=self.fig.get_size_inches()[1] * self.fig.dpi)
        else:
            width = (self.tabview.winfo_width() - 18) / self.fig.dpi
            height = (self.tabview.winfo_height()) / self.fig.dpi
            self.fig.set_size_inches(w=width, h=height, forward=True)  # Re-enable dynamic resizing
            self.canvas_widget.pack(fill="both", expand=True) 

        self.canvas.draw()  # Redraw canvas to apply the automatic size

    def on_closing(self):
        try:
            if hasattr(self, "canvas"): self.canvas.get_tk_widget().destroy()
            if hasattr(self, "fig"): plt.close(self.fig)
        except:
            pass
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
    app.state('normal')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()