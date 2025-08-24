import os
import re
from CTkRangeSlider import *
import customtkinter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from PIL import Image

from LaserSim.crystal import Crystal, plot_cross_sections, plot_small_signal_gain, plot_beta_eq, plot_Isat, plot_Fsat
from LaserSim.pump import Pump
from LaserSim.seed import Seed
from LaserSim.seed_CPA import Seed_CPA
from LaserSim.spectral_losses import Spectral_Losses
from LaserSim.amplifier import Amplifier, plot_inversion1D
from LaserSim.utilities import integ, set_plot_params

set_plot_params()

from LaserSim.spectral_losses import test_reflectivity_approximation


version_number = "25/08"
Standard_path = os.path.dirname(os.path.abspath(__file__))
LaserSim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(LaserSim_path)

plt.style.use('default')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Times New Roman')


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Laser Sim."+version_number)
        self.geometry("1280x720")
        self.replot = False
        self.initialize_variables()
        self.initialize_ui_images()
        self.initialize_ui()
        # self.folder_path.insert(0, Standard_path)
        self.toplevel_window = {'Plot Settings': None,
                                'Legend Settings': None}

        self.setup_plot_area()
        self.update_material(self.material_list.get())
        
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
        self.material_plot_functions = {'cross sections': plot_cross_sections, 
                                        'small signal gain': plot_small_signal_gain,
                                        'equilibrium inversion': plot_beta_eq, 
                                        'saturation intensity': plot_Isat, 
                                        'saturation fluence': plot_Fsat
                                        }

        self.ax = None
        self.plot_index = 0
        self.color = "#212121" # toolbar
        self.text_color = "white"

        # line plot settings
        self.moving_average = 1
        
        # Boolean variables
        self.save_plain_image = customtkinter.BooleanVar(value=False)


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
        
        self.plot_button    = App.create_button(frame, text="Plot material", command=self.test_plot, column=0, row=5, pady=(15,5), image=self.img_reset)
        self.material_list  = App.create_Menu(frame, values=self.materials, column=0, row=1, command=self.update_material, width=120, padx = (10,120-30))
        self.temperature_list  = App.create_Menu(frame, values=["300"], column=0, row=1, command=None, width=60, padx = (120+30,10))
        self.material_list  = App.create_Menu(frame, values=self.materials, column=0, row=1, command=self.update_material, width=120, padx = (10,120-30))
        self.material_plot_list  = App.create_Menu(frame, values=list(self.material_plot_functions.keys()), column=0, row=2, command=None)
        
        self.prev_button    = App.create_button(frame,                      command=None, column=0, row=6, width=90, padx = (10,120), image=self.img_previous)
        self.next_button    = App.create_button(frame,                      command=None, column=0, row=6, width=90, padx = (120,10), image=self.img_next)
        self.set_button     = App.create_button(frame, text="Plot settings",command=None, column=0, row=16, image=self.img_settings)
        self.save_button    = App.create_button(frame, text="Save Figure/data", command=None,     column=0, row=17,  image=self.img_save)
        
        #switches
        self.multiplot_button = App.create_switch(frame, text="Multiplot",  command=None,   column=0, row=7, padx=20, pady=(10,5))
        # self.uselabels_button = App.create_switch(frame, text="Use labels", command=None,  column=0, row=8, padx=20)
        self.uselims_button   = App.create_switch(frame, text="Use limits", command=None,  column=0, row=9, padx=20)
        # self.fit_button       = App.create_switch(frame, text="Use fit",    command=None, column=0, row=10, padx=20)
        # self.normalize_button = App.create_switch(frame, text="Normalize",  command=None,    column=0, row=11, padx=20)
        # self.lineout_button   = App.create_switch(frame, text="Lineout",  command=None,    column=0, row=12, padx=20)
        # self.FFT_button       = App.create_switch(frame, text="FFT",  command=None,    column=0, row=13, padx=20)
        # self.subfolder_button = App.create_switch(frame, text="include subfolders", command=None, row=16, column=0, padx=20)

        self.load_settings_frame()

    # initialize all widgets on the settings frame
    def load_settings_frame(self):
        self.settings_frame = customtkinter.CTkFrame(self, width=1, height=600, corner_radius=0)
        self.settings_frame.grid(row=0, column=4, rowspan=999, sticky="nesw")
        self.settings_frame.grid_columnconfigure(0, minsize=60)
        self.columnconfigure(2,weight=1)


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

    def test_plot(self):
        self.ax.clear()
        print(self.material_list.get())
        crystal = Crystal(material=self.material_list.get(), temperature=int(self.temperature_list.get()))

        plot_function = self.material_plot_functions[self.material_plot_list.get()]
        plot_function(crystal, axis=self.ax)
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
    def reinsert(self, index, text):
        self.delete(0, 'end')  # Delete the current text
        self.insert(index, text)  # Insert the new text

if __name__ == "__main__":

    app = App()
    app.state('zoomed')
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()