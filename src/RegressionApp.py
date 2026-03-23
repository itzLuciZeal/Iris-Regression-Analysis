from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
from PIL import Image, ImageTk
import linear_regression as lnr
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
import numpy as np

src_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(src_dir)
iris_path = os.path.join(BASE_DIR, "Iris.csv")
study_hours_path = os.path.join(BASE_DIR, "Study_Hours.csv")
icon_path = os.path.join(BASE_DIR, "graph.png")

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral', 'DejaVu Serif']

class RegressionApp:
    def __init__(self, app: tk.Tk, dataframe: pd.DataFrame, _index_col: str, set_x: str=None, set_y: str=None, title: str="Regression Analysis", size_alpha: tuple=(30, .6), scatter_col: list[str]=None, line_plot_col: list[str]=None):
        ''' app -> must be in a tk.Tk form | dataframe -> must be in a pd.DataFrame form | _index_col -> indexing used for assigning in each rows | set_x -> any variables that you want to be set as the initial
        set_y -> any variables you want to set as the initial | title -> title of the overall graph | size_alpha -> tuple form with (size, alpha) | scatter_col -> list form with series of strings | line_plot_col -> list form with series of strings '''

        if not isinstance(dataframe, pd.DataFrame): raise TypeError("data frame must be in a pandas Data Frame form")
        if not isinstance(title, str): raise TypeError("Title must be in a string form")
        if (isinstance(size_alpha, tuple) and len(size_alpha) == 1) or isinstance(size_alpha, int):
            self.size = 30
            self.alpha = .6
        if isinstance(size_alpha, tuple) and len(size_alpha) == 2:
            self.size, self.alpha = size_alpha
        if scatter_col is not None and not isinstance(scatter_col, list): raise TypeError("scatter_col must be in a form of list type with series of string in it")
        if line_plot_col is not None and not isinstance(line_plot_col, list): raise TypeError("line_plot_col must be in a form of list type with series of string in it")

        self.icon_image = ImageTk.PhotoImage(file=icon_path)
        self.df = dataframe
        self._index_col = _index_col
        self.df_xyArgs = self.df.columns
        self.species = np.unique(self.df.index)
        self.filtered_df = self.df[self.df.index.isin(self.species)]
        if set_x in self.df_xyArgs and set_y in self.df_xyArgs:
            self.x_arg, self.y_arg = set_x, set_y
        else:
            self.x_arg, self.y_arg = self.df_xyArgs[0], self.df_xyArgs[1]

        self.bg_color = "white"
        self.available_xyArgs = [
            self.df_xyArgs[n]
            for n in range(len(self.df_xyArgs)) 
            if self.x_arg not in self.df_xyArgs[n] and self.y_arg not in self.df_xyArgs[n]
        ]
        self._og_data = [(gp[self.x_arg], gp[self.y_arg]) for _, gp in self.df.groupby(self._index_col, sort=False)]
        self.data = [(gp[self.x_arg], gp[self.y_arg]) for _, gp in self.filtered_df.groupby(self._index_col, sort=False)]
        self.title = title
        self.scatter_col = scatter_col
        self.line_plot_col = line_plot_col
        self.total_n = len(self.data)
        self.current_pos = 1
        self.combine_state = False
        self.pearsonR_state = False
        self.linear_regression_state = False
        self._sub_window_state = False

        self.app = app
        self.app.iconphoto(False, self.icon_image)
        self.app.title("Regression App")
        self.app.config(width=900, height=900, bg=self.bg_color)
        self.app.resizable(width=False, height=False)

        self.main_frame = tk.Frame(master=self.app)
        self.main_frame.config(bg=self.bg_color)
        self.main_frame.pack(pady=25)

        self.fig, self.axs = plt.subplots()
        self.fig.set_size_inches(14, 6.5)

        self._set_up_ui()
        self._graph()
        self.app.mainloop()

    def _set_up_plot_style(self, state: bool=False, x_label: str=None, y_label: str=None):
        ''' Setting up the plotting style and adding the legend '''
        self.axs.set_title(self.title, fontsize=20, fontweight="bold")
        self.axs.set_ylabel(f"(y) {y_label}", fontsize=16)
        self.axs.set_xlabel(f"(x) {x_label}", fontsize=16)
        self.axs.grid(True, alpha=.3)
        if state: self.axs.legend(loc="center left", bbox_to_anchor=(1.03, .5), fontsize=16)
        self.fig.tight_layout()

    def _set_up_ui(self):
        ''' Setting up the UI that will be used and user interface mode '''
        self.string_pos = tk.StringVar(value=f"Group {self.current_pos} out of {self.total_n}")
        self.combine_str = tk.StringVar(value="Combine") if not self.combine_state else tk.StringVar(value="Separate")
        self.pearsonR_str = tk.StringVar(value="Pearson's R") if not self.pearsonR_state else tk.StringVar(value="No Pearson's R")
        self.linear_reg_str = tk.StringVar(value="Regression Line") if not self.linear_regression_state else tk.StringVar(value="No Regression Line")
        self._current_X_str = tk.StringVar(value=f"(X) {self.x_arg}")
        self._current_Y_str = tk.StringVar(value=f"(Y) {self.y_arg}")
        self.canvas = FigureCanvasTkAgg(self.fig, master = self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(bg=self.bg_color)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame, pack_toolbar=False)
        self.toolbar.config(background=self.bg_color)
        self.toolbar._message_label.config(background=self.bg_color)

        for child in self.toolbar.winfo_children():
            child.config(background=self.bg_color)
            if isinstance(child, tk.Button):
                child.config(activebackground=self.bg_color)

        self.toolbar.update()
        self.canvas_widget.pack(padx=30)
        self.toolbar.pack(anchor="center", fill=tk.X)

        self.sec_frame = tk.Frame(master=self.app)
        self.sec_frame.config(bg=self.bg_color)
        self.sec_frame.pack(pady=10)
        self._text_group_no = tk.Label(master=self.sec_frame, textvariable=self.string_pos, font=("Latin Modern Roman", 14, "italic"),
                                       bg=self.bg_color, fg="gray")
        self._text_group_no.pack(side="left", padx=5)
        self._back_btn = tk.Button(master=self.sec_frame, text="<", font=("Latin Modern Roman", 14, "bold"), width=3,
                                   relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                   command=self._move_back)
        self._back_btn.pack(side="left", padx=5)
        self._forward_btn = tk.Button(master=self.sec_frame, text=">", font=("Latin Modern Roman", 14, "bold"), width=3,
                                      relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                      command=self._move_forward)
        self._forward_btn.pack(side="left", padx=5)

        self._combine_btn = tk.Button(master=self.sec_frame, textvariable=self.combine_str, font=("Latin Modern Roman", 14, "bold"), padx=10,
                                      relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                      command=self._grouped_graph)
        self._combine_btn.pack(side="left", padx=5)
        self._with_r = tk.Button(master=self.sec_frame, textvariable=self.pearsonR_str, font=("Latin Modern Roman", 14, "bold"), padx=10,
                                 relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                 command=self._toggle_pearsons_r)
        self._with_r.pack(side="left", padx=5)
        self._with_regLine = tk.Button(master=self.sec_frame, textvariable=self.linear_reg_str, font=("Latin Modern Roman", 14, "bold"), padx=10,
                                       relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                       command=self._toggle_regression_line)
        self._with_regLine.pack(side="left", padx=5)
        self._submit_species_entry = tk.Button(master=self.sec_frame, text="Set Species", font=("Latin Modern Roman", 14, "bold"), padx=5,
                                               relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                               command=self._set_Species)
        self._submit_species_entry.pack(side="left", padx=5)
        self._reset_species_btn = tk.Button(master=self.sec_frame, text="Reset Species", font=("Latin Modern Roman", 14, "bold"), padx=5,
                                               relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                               command=self._reset_Species)
        self._reset_species_btn.pack(side="left", padx=5)
        self._interchange_x_and_y = tk.Button(master=self.sec_frame, text="Interchange X and Y", font=("Latin Modern Roman", 14, "bold"), padx=5,
                                   relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                   command=self._set_Interchange)
        self._interchange_x_and_y.pack(side="left", padx=10)
        
        self.third_frame = tk.Frame(master=self.app)
        self.third_frame.config(bg=self.bg_color)
        self.third_frame.pack(side="bottom", padx=20, pady=25)
        
        self._current_X = tk.Label(master=self.third_frame, textvariable=self._current_X_str, font=("Latin Modern Roman", 14, "italic"),
                                       bg=self.bg_color, fg="gray")
        self._current_X.pack(side="left", padx=5)
        self._x_comboBox = ttk.Combobox(master=self.third_frame, values=self.available_xyArgs, font=("Latin Modern Roman", 14), state="readonly")
        self._x_comboBox.pack(side="left", padx=10)
        self._submit_x = tk.Button(master=self.third_frame, text="Set X", font=("Latin Modern Roman", 14, "bold"), padx=5,
                                   relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                   command=self._set_X)
        self._submit_x.pack(side="left", padx=5)
        
        self._current_Y = tk.Label(master=self.third_frame, textvariable=self._current_Y_str, font=("Latin Modern Roman", 14, "italic"),
                                       bg=self.bg_color, fg="gray")
        self._current_Y.pack(side="left", padx=5)
        self._y_comboBox = ttk.Combobox(master=self.third_frame, values=self.available_xyArgs, font=("Latin Modern Roman", 14), state="readonly")
        self._y_comboBox.pack(side="left", padx=10)
        self._submit_y = tk.Button(master=self.third_frame, text="Set Y", font=("Latin Modern Roman", 14, "bold"), padx=5,
                                   relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                   command=self._set_Y)
        self._submit_y.pack(side="left", padx=5)

        # Checking before activating fully
        if len(self.available_xyArgs) != 0: 
            self._x_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0]))
            self._y_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0]))

        if len(self.df_xyArgs) == 2:
            self._x_comboBox.config(state="disabled")
            self._y_comboBox.config(state="disabled")
            self._submit_x.config(state="disabled")
            self._submit_y.config(state="disabled")
        
    def _draw_data(self, indices):
        ''' Internal helper to plot specific indices '''
        self.axs.clear()
        x_desc, y_desc = "", ""

        for i in indices:
            x, y = self.data[i][0], self.data[i][1]
            x_desc, y_desc = x.name, y.name
            current_name = np.unique(x.index)[0]
            if self.scatter_col is not None:
                available_scatter_color = [
                    self.scatter_col[n]
                    for n in range(len(self.scatter_col))
                    if self.scatter_col[n] is not None and isinstance(self.scatter_col[n], str)
                ]
            else:
                available_scatter_color = None

            if self.line_plot_col is not None:
                available_line_color = [
                    self.line_plot_col[n]
                    for n in range(len(self.line_plot_col))
                    if self.line_plot_col[n] is not None and isinstance(self.line_plot_col[n], str)
                ]
            else:
                available_line_color = None

            try:
                if self.pearsonR_state: # Adding Pearson R to the Legend
                    r_val = lnr.Inference(x, y).pearson_r()
                    try:
                        if available_scatter_color is not None: self.axs.scatter(x, y, color=available_scatter_color[i], s=self.size, alpha=self.alpha, label=f"{current_name} | r = {r_val:.2f}")
                        else: self.axs.scatter(x, y, s=self.size, alpha=self.alpha, label=f"{current_name} | r = {r_val:.2f}")
                    except IndexError:
                        self.axs.scatter(x, y, s=self.size, alpha=self.alpha, label=f"{current_name} | r = {r_val:.2f}")
                else:
                    if available_scatter_color is not None: self.axs.scatter(x, y, color=available_scatter_color[i], s=self.size, alpha=self.alpha, label=current_name)
                    else: self.axs.scatter(x, y, s=self.size, alpha=self.alpha, label=current_name)
            except IndexError:
                if self.pearsonR_state:
                    r_val = lnr.Inference(x, y).pearson_r()
                    self.axs.scatter(x, y, s=self.size, alpha=self.alpha, label=f"{current_name} | r = {r_val:.2f}")
                else: self.axs.scatter(x, y, s=self.size, alpha=self.alpha, label=current_name)
            
            if self.linear_regression_state: # Adding Linear Regression to the Legend
                x_lin, y_lin = lnr.Inference(x, y).reg_line_array()
                try:
                    if available_line_color is not None: self.axs.plot(x_lin, y_lin, color=available_line_color[i], linewidth=2.5, label=f"({current_name}) {lnr.Inference(x, y).linear_equation_text()}")
                    else: self.axs.plot(x_lin, y_lin, linewidth=2.5, label=f"({current_name}) {lnr.Inference(x, y).linear_equation_text()}")
                except IndexError:
                    self.axs.plot(x_lin, y_lin, linewidth=2.5, label=f"({current_name}) {lnr.Inference(x, y).linear_equation_text()}")

        self._set_up_plot_style(state=True, x_label=x_desc, y_label=y_desc)
        self.canvas.draw()
    
    def _move_back(self):
        ''' Toggle to move backward from the current position in single graph '''
        if self.current_pos == 1:
            self.current_pos = self.total_n
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            self._graph()
        
        else:
            self.current_pos = self.current_pos - 1
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            self._graph()
        
    def _move_forward(self):
        ''' Toggle to move forward from the current position in single graph '''
        if self.current_pos == self.total_n:
            self.current_pos = 1
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            self._graph()

        else:
            self.current_pos = self.current_pos + 1
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            self._graph()

    def _graph(self):
        ''' Plots only the active group with saved current position '''
        self._draw_data([self.current_pos - 1])
    
    def _grouped_graph(self):
        ''' Toggle between single group to all groups '''
        self.combine_state = not self.combine_state

        if self.combine_state:
            # Show All
            self._draw_data(range(self.total_n))
            self._back_btn.config(state="disabled")
            self._forward_btn.config(state="disabled")
            self.string_pos.set("Combine In Use")
            self.combine_str.set("Separate")
        else:
            # Show Single
            self._graph()
            self._back_btn.config(state="normal")
            self._forward_btn.config(state="normal")
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            self.combine_str.set("Combine")
    
    def _available_xyArgs(self):
        ''' Updates the available x and y variable data to be used '''
        self.available_xyArgs = [
            self.df_xyArgs[n] 
            for n in range(len(self.df_xyArgs)) 
            if self.x_arg not in self.df_xyArgs[n] and self.y_arg not in self.df_xyArgs[n]
        ]
        
    def _set_X(self):
        ''' Toggle to set a new X variable data '''
        new_set_X = self._x_comboBox.get()
        self.x_arg = new_set_X
        self._available_xyArgs()
        self._current_X_str.set(f"(X) {self.x_arg}")
        self._x_comboBox.config(values=self.available_xyArgs)
        self._y_comboBox.config(values=self.available_xyArgs)
        if len(self.available_xyArgs) != 0: self._x_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0]))
        if len(self.available_xyArgs) != 0: self._y_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0]))
        self.data = [(gp[self.x_arg], gp[self.y_arg]) for _, gp in self.filtered_df.groupby(self._index_col, sort=False)]

        if self.combine_state:
            self._draw_data(range(self.total_n))     
        else:
            self._graph()

    def _set_Y(self):
        ''' Toggle to set a new Y variable data '''
        new_set_Y = self._y_comboBox.get()
        self.y_arg = new_set_Y
        self._available_xyArgs()
        self._current_Y_str.set(f"(Y) {self.y_arg}")
        self._x_comboBox.config(values=self.available_xyArgs)
        self._y_comboBox.config(values=self.available_xyArgs)
        if len(self.available_xyArgs) != 0: self._x_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0]))
        if len(self.available_xyArgs) != 0: self._y_comboBox.current(self.available_xyArgs.index(self.available_xyArgs[0])) 
        self.data = [(gp[self.x_arg], gp[self.y_arg]) for _, gp in self.filtered_df.groupby(self._index_col, sort=False)]

        if self.combine_state:
            self._draw_data(range(self.total_n))     
        else:
            self._graph()
    
    def _set_Interchange(self):
        ''' Toggle to interchange the places of X and Y variable data '''
        self.x_arg, self.y_arg = self.y_arg, self.x_arg
        self._current_X_str.set(f"(X) {self.x_arg}")
        self._current_Y_str.set(f"(Y) {self.y_arg}")
        self.data = [(gp[self.x_arg], gp[self.y_arg]) for _, gp in self.filtered_df.groupby(self._index_col, sort=False)]

        if self.combine_state:
            self._draw_data(range(self.total_n))
        else:
            self._graph()

    def _set_Species(self):
        ''' Toggle to set specific species to be analyzed '''

        if not self._sub_window_state:
            self._set_Species_window = tk.Toplevel()
            self._set_Species_window.config(bg="white")
            self._set_Species_window.iconphoto(False, self.icon_image)
            self._set_Species_window.title("Set Species")
            self._set_Species_window.geometry("370x420")
            self._set_Species_window.protocol("WM_DELETE_WINDOW", lambda: None)
            window_width = 370
            window_height = 420
            screen_width = self._set_Species_window.winfo_screenwidth()
            screen_height = self._set_Species_window.winfo_screenheight()
            center_x = int((screen_width / 2) - (window_width / 2))
            center_y = int((screen_height / 2) - (window_height / 2))
            self._set_Species_window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
            self._set_Species_window.resizable(width=False, height=False)
            self._title = tk.Label(master=self._set_Species_window, text="Set Specific Species", font=("Latin Modern Roman", 16, "bold"), bg=self.bg_color)
            self._title.pack(pady=10)

            self.container = tk.Frame(self._set_Species_window, bg=self.bg_color)
            self.container.pack(fill="both", expand=True)

            self._species_canvas = tk.Canvas(self.container, bg=self.bg_color, highlightthickness=0)
            self._species_canvas_scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self._species_canvas.yview)

            self._scrollable_frame = tk.Frame(self._species_canvas, bg=self.bg_color)
            self._scrollable_frame.bind(
                "<Configure>",
                lambda e: self._species_canvas.configure(scrollregion=self._species_canvas.bbox("all"))
            )

            self._species_canvas.create_window((0, 0), window=self._scrollable_frame, anchor="nw")
            self._species_canvas.configure(yscrollcommand=self._species_canvas_scrollbar.set)

            self._species_canvas.pack(side="left", fill="both", expand=True)
            self._species_canvas_scrollbar.pack(side="right", fill="y")

            # Update scrollregion whenever the interior frame changes size
            self._scrollable_frame.bind(
                "<Configure>",
                lambda e: self._species_canvas.configure(scrollregion=self._species_canvas.bbox("all"))
            )
            self._species_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

            n = len(np.unique(self.df.index))
            self.check_vars = []
            self.check_buttons = []

            for i in range(n): # Creating checkbuttons for each species
                var = tk.StringVar()
                self.check_vars.append(var)
                current_name = np.unique(self._og_data[i][0].index)[0]

                cb = tk.Checkbutton(
                    self._scrollable_frame, 
                    text=current_name, 
                    variable=var,
                    onvalue=current_name,
                    offvalue="",
                    bg=self.bg_color,
                    activebackground=self.bg_color, # Keeps it same color when clicked
                    font=("Latin Modern Roman", 14, "italic")
                )
                cb.pack(anchor="w", padx=50, pady=4)
                self.check_buttons.append(cb)
            
            self._submit_btn = tk.Button(master=self._set_Species_window, text="Submit Species", font=("Latin Modern Roman", 12, "bold"), padx=5,
                                               relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                               command=self._submit_Species)
            self._submit_btn.pack(side="left", padx=20, pady=20)
            self._exit_btn = tk.Button(master=self._set_Species_window, text="Return to Graph", font=("Latin Modern Roman", 12, "bold"), padx=5,
                                               relief="ridge", bg=self.bg_color, fg="black", activebackground="whitesmoke", activeforeground="black",
                                               command=self._exit_set_Species)
            self._exit_btn.pack(side="left", padx=20, pady=20)
            self._sub_window_state = not self._sub_window_state
        
    def _submit_Species(self):
        ''' Toggle to submit the species and update the graph '''
        available_species = np.unique(self.df.index)
        species_list = [self.check_vars[each_var].get() for each_var in range(len(self.check_vars))]
        valid_species = [s for s in species_list if s in available_species]

        if valid_species:
            self.species = valid_species
            self.filtered_df = self.df[self.df.index.isin(self.species)]
            self.data = [(gp[self.x_arg], gp[self.y_arg]) for name, gp in self.filtered_df.groupby(self._index_col, sort=False)]
            self.total_n = len(self.data)
            self.current_pos = 1 if self.current_pos > self.total_n else self.current_pos
            
            if self.combine_state:
                self._draw_data(range(self.total_n))
            
            else:
                self._graph()
                self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")

    def _exit_set_Species(self):
        ''' Toggle to exit the set species window and return to the main graph'''
        self._set_Species_window.destroy()
        self._sub_window_state = not self._sub_window_state
                
    def _reset_Species(self):
        ''' Toggle to reset the number of species to normal '''
        available_species = np.unique(self.df.index)
        self.species = available_species
        self.filtered_df = self.df[self.df.index.isin(self.species)]
        self.data = [(gp[self.x_arg], gp[self.y_arg]) for name, gp in self.filtered_df.groupby(self._index_col, sort=False)]
        self.total_n = len(self.data)
        self.current_pos = 1 if self.current_pos > self.total_n else self.current_pos
        
        if self.combine_state:
            self._draw_data(range(self.total_n))
        
        else:
            self._graph()
            self.string_pos.set(f"Group {self.current_pos} out of {self.total_n}")
            
    def _toggle_pearsons_r(self):
        ''' Toggle pearson r '''
        self.pearsonR_state = not self.pearsonR_state
    
        # Update the pearson r button
        self.pearsonR_str.set("No Pearson's R" if self.pearsonR_state else "Pearson's R")
        
        # Use the helper to refresh the view
        if self.combine_state:
            self._draw_data(range(self.total_n))  # Redraw all with/without R
        else:
            self._graph()  # Redraw current single with/without R
    
    def _toggle_regression_line(self):
        ''' Toggle linear regression line '''
        self.linear_regression_state = not self.linear_regression_state

        # Update the linear regression button
        self.linear_reg_str.set("No Regression Line" if self.linear_regression_state else "Regression Line")

        # Use the helpter to refresh the view
        if self.combine_state:
            self._draw_data(range(self.total_n)) # Redraw all with/without linear regression line
        else:
            self._graph() # Redraw current single with/without linear regression line

    def _on_mousewheel(self, event):
        # On Windows, event.delta is typically 120 or -120
        # Dividing by -120 makes it scroll 1 unit per "click" of the wheel
        self._species_canvas.yview_scroll(int(-1*(event.delta/120)), "units")


df_Iris = pd.read_csv(iris_path, index_col="Species").drop(columns="Id")
df_Learners = pd.read_csv(study_hours_path, index_col="Learners")

scatter_colors = ["red", "royalblue", "rebeccapurple"]
line_colors = ["black", "midnightblue", "gold"]

app = tk.Tk()
regApp = RegressionApp(app=app, dataframe=df_Iris, _index_col="Species", title="Iris Regression Analysis", size_alpha=(35, .7), scatter_col=scatter_colors, line_plot_col=line_colors)