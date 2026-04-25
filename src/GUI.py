import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import cm
from matplotlib.colors import Normalize
from pyproj import Transformer
import matplotlib.image as mpimg
from src.API_caller import download_dem_for_samples
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from src.optimiser import optimise_candidates, scores_to_dataframe, OptimiserWeights

tif_path = None
metadata_csv_path = None
loaded_sample_metadata = []

def start_gui(run_program): #entry point for the program.

    def set_coordinate_entries(lon, lat):
            lon_text = f"{lon:.6f}"
            lat_text = f"{lat:.6f}"

            lon_var.set(lon_text)
            lat_var.set(lat_text)

            selected = sample_tree.selection()
            if not selected:
                return  # if no row is selected, just fill the editor boxes

            item_id = selected[0]
            values = list(sample_tree.item(item_id, "values"))

            values[1] = lon_text   # longitude column
            values[2] = lat_text   # latitude column

            sample_tree.item(item_id, values=values)

    def add_scale_bar(ax, length_m: float) -> None:
        x0, x1 = ax.get_xlim() #retrieve x limits of axes.
        y0, y1 = ax.get_ylim() #retrieve y limits of axes.

        x = x0 + (x1 - x0) * 0.07
        y = y0 + (y1 - y0) * 0.07 #place bar 7% up and to the right from the bottom left corner.

        ax.plot([x, x + length_m], [y, y], linewidth=4, color="black") #draw a horizontal line.
        label = f"{int(length_m)} m" if length_m < 1000 else f"{length_m / 1000:.1f} km" #label line in metres if below 1km or else kilometres.
        ax.text(
            x + length_m / 2.0,
            y + (y1 - y0) * 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        ) #styling for label.

    class RightSideBar(ttk.Frame): #right side bar showing the DEM preview and overlay result, ttk.Frame specifies it as a widget container.
        def __init__(self, parent):
            super().__init__(parent, padding=8) #initialise the object as a ttk.Frame and add padding.


            self.dem = None #store DEM data.
            self.dem_transform = None #store raster transformer.
            self.dem_crs = None #store raster CRS.
            self.dem_path = None #store file path to DEM.
            self.count_overlay = None #store frequency count result.
            self.observer_points_xy = [] #store observer cooridnates.

            self.point_selected_callback = None #reference to helper function for auto filling boxes on click.
            self.clicked_points = [] #store the coordinates the user has clicked on.
            self.tip = None

            self.rowconfigure(1, weight=1) #only allow preview to grow if needed.
            self.columnconfigure(0, weight=1) #allow preview section to be streched sideways.

            self.title_label = ttk.Label(self, text="DEM PREVIEW", font=("Segoe UI", 12, "bold")) #add Title.
            self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6)) #position Title.

            self.fig = Figure(figsize=(11, 5.8), dpi=100) #create a Matplotlib Figure object.
            self.ax_count = self.fig.add_subplot(111) #add axes.
            self.ax_count.set_title("No DEM loaded") #if no DEM loaded, inform user.

            self.scale_bar_length_m = None #store scale bar length for result view.

            self.ax_count.set_xticks([])
            self.ax_count.set_yticks([]) #do not show ticks around empty canvas message.

            self.canvas = FigureCanvasTkAgg(self.fig, master=self) #create a Tkinter compatible canvas wrapper for Matplotlib.
            self.canvas_widget = self.canvas.get_tk_widget() #render it with equivalent widget corresponding to a Matplotlib figure.
            self.canvas_widget.grid(row=1, column=0, sticky="nsew") #force DEM canvas to fill entire grid cell.

            self.toolbar_frame = ttk.Frame(self) #add container for Matplotlib toolbar.
            self.toolbar_frame.grid(row=2, column=0, sticky="ew") #position Matplotlib toolbar.
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False) #create the Matplotlib toolbar to add interactivity to the canvas.
            self.toolbar.update() #standard practice: refresh toolbar before displaying.
            self.toolbar.pack(side="left") #move toolbar to the left.
            
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.canvas.mpl_connect("scroll_event", self.on_scroll)

            self.show_overlay = tk.BooleanVar(value=True) #create a Boolean state for whether overlay should be visible, default to show.
            self.count_overlay_im = None #default to no overlay image.
            self.count_overlay_cbar = None #default to no colour bar.

            self.view_xlim = None
            self.view_ylim = None #store information about current manual zoom so it isn't lost on refresh.
            self.view_extent = None #store automatic zoom calculated by programw for the current result.

            menubar = tk.Menu(parent) #create menu for the window in the top bar...
            parent.config(menu=menubar) #...and attach it.

            file_menu = tk.Menu(menubar, tearoff=0) #create file menu inside the top bar...
            menubar.add_cascade(label="File", menu=file_menu) #...and add it.
            file_menu.add_command(label="Load metadata CSV", command=load_metadata_file) #create a button to load metadata from CSV.

            view_menu = tk.Menu(menubar, tearoff=0) #create view menu inside the top bar...
            menubar.add_cascade(label="View", menu=view_menu) #...and add it.

            view_menu.add_checkbutton(
                label="Show overlay",
                variable=self.show_overlay,
                command=self.toggle_overlay
            ) #create a button to toggle overlay.

            self.canvas.mpl_connect("button_press_event", self.on_click) #create a click handler to detect what coordinates a user may click on when drawing a polygon.


  
        def on_scroll(self, event): #when the user scrolls mouse wheel over the plot.
            if event.inaxes != self.ax_count:
                return #if scroll happens outside the plot, do nothing.

            if event.xdata is None or event.ydata is None:
                return #if graph coordinates cannot be worked out, do nothing.

            if self.toolbar.mode != "":
                return #if a Matplotlib tool is active, do not interfere.

            xdata = event.xdata #x position of mouse in data coordinates.
            ydata = event.ydata #y position of mouse in data coordinates.

            cur_xlim = self.ax_count.get_xlim() #retrieve current x axis limits.
            cur_ylim = self.ax_count.get_ylim() #retrieve current y axis limits.

            x_left = xdata - cur_xlim[0] #distance from mouse to left edge.
            x_right = cur_xlim[1] - xdata #distance from mouse to right edge.
            y_bottom = ydata - cur_ylim[0] #distance from mouse to bottom edge.
            y_top = cur_ylim[1] - ydata #distance from mouse to top edge.

            if event.button == "up":
                scale_factor = 0.8 #zoom in.
            elif event.button == "down":
                scale_factor = 1.25 #zoom out.
            else:
                return #if unknown scroll direction, do nothing.

            new_xlim = [xdata - x_left * scale_factor, xdata + x_right * scale_factor]
            new_ylim = [ydata - y_bottom * scale_factor, ydata + y_top * scale_factor] #scale limits around the mouse position.

            self.ax_count.set_xlim(new_xlim)
            self.ax_count.set_ylim(new_ylim)

            self.view_xlim = self.ax_count.get_xlim()
            self.view_ylim = self.ax_count.get_ylim() #store manual zoom so redraws keep it.

            self.canvas.draw_idle() #refresh canvas.    

        def format_axes_nicely(self, ax) -> None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

            x_formatter = ScalarFormatter(useMathText=True)
            y_formatter = ScalarFormatter(useMathText=True)

            x_formatter.set_scientific(True)
            y_formatter.set_scientific(True)

            x_formatter.set_powerlimits((0, 0))
            y_formatter.set_powerlimits((0, 0))

            ax.xaxis.set_major_formatter(x_formatter)
            ax.yaxis.set_major_formatter(y_formatter)

            ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)

        def on_click(self, event): #when the user clicks mouse.
            if self.dem is None:
                return
            if event.inaxes != self.ax_count:
                return #if click happens outside of canvas, do nothing.
            if event.xdata is None or event.ydata is None:
                return #if the graph coordinates cannot be worked out, do nothing.
            if self.toolbar.mode != "":
                return

            x = event.xdata #obtain x coordinate user clicked on.
            y = event.ydata #obtain y coordinate user clicked on.
            self.clicked_points.append((x, y)) #return coordinates user clicked on.

            if self.dem_crs is not None and self.point_selected_callback is not None: #if a coordinate system can be found, and the helper function is ready.
                transformer = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True) #create DEM transformer.
                lon, lat = transformer.transform(x, y) #convert clicked position into longitude and latitude.
                self.point_selected_callback(lon, lat) #send coordinates to helper function.

            self._redraw()

        def load_dem(self, dem_path):
            with rasterio.open(dem_path) as src: #open GeoTIFF file.
                arr = src.read(1).astype(np.float64) #read elevation raster band.
                nodata = src.nodata #value that raster uses for "nodata".

                if nodata is not None: #if there are cells that have no data.
                    arr = np.ma.masked_equal(arr, nodata) #mask each one.

                arr = np.ma.masked_invalid(arr) #mask invalid cells.       
                arr = np.ma.masked_where(np.abs(arr) > 1e20, arr) #mask cells with an absurdly large size.

                self.dem = arr #store DEM array in GUI.
                self.dem_transform = src.transform #store affine transform.
                self.dem_crs = src.crs #store raster coordinate system.
                self.dem_path = dem_path #store DEM path.

            self.count_overlay = None #clear previous DEM overlay.
            self.observer_points_xy = [] #clear previous observer points.
            self.scale_bar_length_m = None #clear previous scale bar.
            self._redraw() #redraw DEM.

        def set_results(self, count_overlay, observer_points=None, view_extent=None, scale_bar_length_m=None):
            if self.dem is None:
                raise ValueError("Load a DEM before setting an overlay.") #check a DEM file is present.

            self.count_overlay = count_overlay #store DEM LoS render.
            self.observer_points_xy = observer_points if observer_points is not None else [] #store observer coordinates.
            self.view_extent = view_extent #store automatic zoom for the current result.
            self.scale_bar_length_m = scale_bar_length_m #store scale bar length for the current result.
            self.view_xlim = None
            self.view_ylim = None #reset manual zoom so a fresh submit uses the program's automatic zoom.
            self._redraw() #render.

        def toggle_overlay(self):
            self._redraw()
       
        def clear_overlay(self):
            self.count_overlay = None #remove any previous overlay.
            self.observer_points_xy = [] #remove any previous observer point.
            self.scale_bar_length_m = None #remove previous scale bar information.
            self.view_extent = None #remove previous automatic zoom information.
            self._redraw() #render preview area again.

        def _remove_colourbars(self):

            if self.count_overlay_cbar is not None:
                self.count_overlay_cbar.remove()
                self.count_overlay_cbar = None

        def _redraw(self):
            self._remove_colourbars()
            self.ax_count.clear() #clean axes to prevent buildup.

            if self.dem is None: #if no DEM given, show blank DEM message.
                self.ax_count.set_title("No DEM loaded")
                self.ax_count.set_xticks([])
                self.ax_count.set_yticks([])
                self.canvas.draw_idle()
                return

            if self.count_overlay is not None and self.show_overlay.get() and self.view_extent is not None: #if a result exists and overlays are enabled.
                left, right, bottom, top = self.view_extent

                count_vmax = max(1, int(np.max(self.count_overlay))) #find biggest count to scale colourbar.

                self.count_overlay_im = self.ax_count.imshow(
                    self.count_overlay,
                    extent=(left, right, bottom, top),
                    origin="upper",
                    cmap="viridis",
                    vmin=0,
                    vmax=count_vmax,
                ) #render the frequency raster exactly like visibility_frequency.py.

                self.count_overlay_cbar = self.fig.colorbar(self.count_overlay_im, ax=self.ax_count)
                self.count_overlay_cbar.set_label("Number of observers seeing each cell")

                self.ax_count.set_title("Visibility frequency")
                self.ax_count.set_xlabel("X")
                self.ax_count.set_ylabel("Y")
                self.ax_count.set_aspect("equal")
                self.format_axes_nicely(self.ax_count)

                if self.scale_bar_length_m is not None:
                    add_scale_bar(self.ax_count, self.scale_bar_length_m) #draw scale bar exactly like visibility_frequency.py.

            else:
                show(
                    self.dem,
                    transform=self.dem_transform,
                    ax=self.ax_count,
                    cmap="terrain"
                )  #render DEM base image.

                self.ax_count.set_title("DEM PREVIEW")
                self.ax_count.set_xlabel("Easting (m)")
                self.ax_count.set_ylabel("Northing (m)")
                self.count_overlay_im = None

            if self.view_xlim is not None and self.view_ylim is not None:
                self.ax_count.set_xlim(self.view_xlim)
                self.ax_count.set_ylim(self.view_ylim) #redraw zoom based on saved information.

            self.fig.tight_layout()
            self.canvas.draw_idle()

        def hide_tip(self, event=None):
            if self.tip is not None:
                self.tip.destroy() #remove window.
                self.tip = None #...and the reference to it.
                
    class LeftSideBar: #for each helper button

            def __init__(self, widget, text):
                self.widget = widget #widget the popup belongs to.
                self.text = text #text that should appear in popup.
                self.tip = None #start with no text showing.
                
                self.widget.bind("<Enter>", self.show_tip) #show text when mouse enters widget.
                self.widget.bind("<Leave>", self.hide_tip) #hide text when mouse leaves widget.
        
            def show_tip(self, event=None): #when mouse enters help widget.
                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() + 20 #find coordinates of "?" icon, then move 20 right and up.

                self.tip = tw = tk.Toplevel(self.widget) #create a top level window for the popup.
                tw.wm_overrideredirect(True) #specifies this is a blank window.
                tw.wm_geometry(f"+{x}+{y}") #shift window to avoid overlapping with "?".

                label = tk.Label(
                    tw,
                    text=self.text,
                    justify="left",
                    background="#ffffe0",
                    relief="solid",
                    borderwidth=1,
                    padx=6,
                    pady=4
                ) #customise appearance of label inside window.
                label.pack() #place label inside tooltip window.

            def hide_tip(self, event=None):
                    if self.tip is not None:
                        self.tip.destroy() #remove window.
                        self.tip = None #...and the reference to it.


    def populate_sample_entries(samples):
        sample_tree.delete(*sample_tree.get_children())  # clear old rows

        for i, sample in enumerate(samples, start=1):
            sample_tree.insert(
                "",
                "end",
                values=(
                    i,
                    sample["lon"],
                    sample["lat"],
                    sample["observer_height"],
                    sample["heading_deg"],
                ),
            )

        children = sample_tree.get_children()
        if children:
            sample_tree.selection_set(children[0])
            sample_tree.focus(children[0])
            on_tree_select()

        update_left_scrollregion()

    def load_metadata_csv(file_path):
        samples = [] #will store sample data.

        with open(file_path, newline="", encoding="utf-8-sig") as csvfile: #open CSV.
            reader = csv.DictReader(csvfile) #read in data as a dictionary.

            required_columns = {"lon", "lat", "observer_height", "heading_deg"} #assert which columns are needed.
            if reader.fieldnames is None:
                raise ValueError("The CSV file has no header row.") #if there is no header row, raise an error.

            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing))}") #if there are any missing columns, raise an error.

            for row_index, row in enumerate(reader, start=2): #start at 2 because row 1 is the header.
                try:
                    lon = float(row["lon"])
                    lat = float(row["lat"])
                    observer_height = float(row["observer_height"]) #extract longitude, latitutde and observer height data.
                    heading_deg = float(row["heading_deg"]) #extract heading data.
                except ValueError:
                    raise ValueError(f"Invalid numeric value in CSV row {row_index}.") #if any data is not a number, raise an error.

                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude out of range in CSV row {row_index}.") #if any longitude is out of range, raise an error.

                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude out of range in CSV row {row_index}.") #if any latitude is out of range, raise an error.

                if observer_height <= 0:
                    raise ValueError(f"Observer height must be greater than 0 in CSV row {row_index}.") #if height is 0 or less, raise an error.

                if not (0 <= heading_deg < 360):
                    raise ValueError(f"Heading must be between 0 and 360 degrees in CSV row {row_index}.") #if heading is out of range, raise an error.

                samples.append({
                    "lon": lon,
                    "lat": lat,
                    "observer_height": observer_height,
                    "heading_deg": heading_deg
                }) #add data to samples.

        return samples
    
    def metadata_handler(): 
        selected_metadata_var = tk.StringVar(value=metadata_csv_path if metadata_csv_path else "No metadata CSV selected") #Tkinter text displaying the current CSV path or "No metadata CSV selected".

        def load_metadata_file(): 
            global metadata_csv_path 
            global loaded_sample_metadata 

            file_path = filedialog.askopenfilename( #open a file picker window and store the chosen file path in file_path.
                parent=root, #pop up must be shown in the app.
                title="Open metadata CSV for VISTA", #text at the top of window.
                initialdir=".", #begin browsing in the directory of the program.
                filetypes=[
                    ("CSV files", "*.csv"), #only show files ending in .csv as the type we want the user to pick.
                ] #files that are shown as acceptable, in this case CSV.
            ) #use file explorer to allow the user to select a file.

            if not file_path: #check whether the user cancelled the file picker instead of selecting a file.
                return  #if user aborts, stop function.

            try:
                samples = load_metadata_csv(file_path) #read the CSV file and turn it into a validated list of sample dictionaries.
                print("Rows loaded from CSV:", len(samples))
                metadata_csv_path = file_path #store the chosen CSV path.
                loaded_sample_metadata = samples #store the loaded sample list.
                selected_metadata_var.set(file_path) #update the Tkinter text variable to display the chosen CSV file path.
                populate_sample_entries(samples) #fill the visible entry boxes in the GUI with the values loaded from the CSV file.
                error_label.config(text="") #clear any old error message because loading succeeded.
            except Exception as e: 
                messagebox.showerror("Load error", f"Could not load metadata CSV:\n{e}") #if there is an error with loading the file, alert the user.

        return selected_metadata_var, load_metadata_file 
   
   
   
    def validate_inputs():
        max_observer_height = 10000
        samples = []

        items = sample_tree.get_children()
        if not items:
            raise ValueError("Please enter at least one sample or load a metadata CSV.")

        for idx, item_id in enumerate(items, start=1):
            values = sample_tree.item(item_id, "values")

            lon_text = str(values[1]).strip()
            lat_text = str(values[2]).strip()
            observer_height_text = str(values[3]).strip()
            heading_text = str(values[4]).strip()

            try:
                lon = float(lon_text)
                lat = float(lat_text)
                observer_height = float(observer_height_text)
                heading_deg = float(heading_text)
            except ValueError:
                raise ValueError(
                    f"Longitude, latitude, observer height, and heading must all be numbers for sample {idx}."
                )

            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude must be between -180 and 180 for sample {idx}.")

            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90 for sample {idx}.")

            if observer_height <= 0:
                raise ValueError(f"Observer height must be greater than 0 metres for sample {idx}.")

            if observer_height > max_observer_height:
                raise ValueError(
                    f"Observer height must not exceed {max_observer_height} metres for sample {idx}."
                )

            if not (0 <= heading_deg < 360):
                raise ValueError(f"Heading must be between 0 and 360 degrees for sample {idx}.")

            samples.append({
                "lon": lon,
                "lat": lat,
                "observer_height": observer_height,
                "heading_deg": heading_deg
            })

        return samples
    
    def validate_max_distance():
        value_text = max_distance_var.get().strip()
        if not value_text:
            raise ValueError("Please enter a maximum distance.")
        try:
            value = float(value_text)
        except ValueError:
            raise ValueError("Maximum distance must be a number.")
        if value <= 0:
            raise ValueError("Maximum distance must be greater than 0 metres.")
        return value


    def get_dem_path(sample_metadata, max_distance):
        if dem_source_var.get() == "local":
            if not tif_path:
                raise ValueError("Please select a GeoTIFF file first.")
            return tif_path

        return download_dem_for_samples(sample_metadata, max_distance)

    def show_error(message):
            error_label.config(text=message) #update the error label with the error text.

    def file_handler():
        
            selected_file_var = tk.StringVar(value=tif_path if tif_path else "No file selected") #default display when no file selected.

            def validate_file(file_path):
                ext = Path(file_path).suffix.lower() #obtain ending of file name.

                if ext not in [".tif", ".tiff"]: #if ending isn't ".tif" or ".tiff"...
                    raise ValueError(f"Unsupported file type: {ext}") #...raise an error.
                    
            def load_file():
                global tif_path
                file_path = filedialog.askopenfilename(
                    parent=root, #pop up must be shown in the app.
                    title="Open file for VISTA", #text at the top of window.
                    initialdir=".", #begin browsing in the directory of the program.
                    filetypes=[
                        ("GeoTIFF files", "*.tif *.tiff"),
                    ] #files that are shown as acceptable, in this case TIFF.
                ) #use file explorer to allow the user to select a file.

                if not file_path:
                    return  #if user aborts, stop function.

                try:
                    validate_file(file_path) #validate file function.
                    print("validate_inputs sample count:", len(loaded_sample_metadata) if loaded_sample_metadata else "manual entry")
                    tif_path = file_path #store chosen file path to be passed into run_program later.
                    selected_file_var.set(file_path)
                    right_sidebar.load_dem(tif_path) #load initial DEM preview.
                    error_label.config(text="")
                except Exception as e:
                    messagebox.showerror("Load error", f"Could not load file:\n{e}") #if there is an error with loading the file, alert the user.

            top_bar = ttk.Frame(root, padding=8) #create a container for the file button and text.
            top_bar.grid(row=0, column=0, columnspan=2, sticky="ew") #stretch horizontally.

            file_button = ttk.Button(top_bar, text="DEM", command=load_file)
            file_button.pack(side="left") #move to the leftmost side.

            file_label = ttk.Label(top_bar, textvariable=selected_file_var) #label for currently selected file, or default text if no file loaded.
            file_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.

            metadata_button = ttk.Button(top_bar, text="CSV", command=load_metadata_file) #create button that runs "load_metadata_file" when clicked.
            metadata_button.pack(side="left", padx=(20, 0)) #move to the leftmost side, next to the file path.

            metadata_label = ttk.Label(top_bar, textvariable=selected_metadata_var) #label for currently selected metadata CSV, or default text if no file loaded.
            metadata_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.

            max_distance_label = ttk.Label(top_bar, text="Max distance (m):")
            max_distance_label.pack(side="left", padx=(20, 4))

            max_distance_entry = ttk.Entry(top_bar, width=10, textvariable=max_distance_var)
            max_distance_entry.pack(side="left")

            dem_mode_label = ttk.Label(top_bar, text="DEM source:")
            dem_mode_label.pack(side="left", padx=(20, 4))

            local_radio = ttk.Radiobutton(
                top_bar,
                text="Use local DEM",
                variable=dem_source_var,
                value="local",
            )
            local_radio.pack(side="left")

            download_radio = ttk.Radiobutton(
                top_bar,
                text="Download DEM from OpenTopography",
                variable=dem_source_var,
                value="download",
            )
            download_radio.pack(side="left", padx=(10, 0))

            
    def submit():
        global tif_path
        error_label.config(text="")
        try:
            sample_metadata = validate_inputs()
            max_distance = validate_max_distance()
            dem_path = get_dem_path(sample_metadata, max_distance)

            right_sidebar.load_dem(dem_path)

            print("Number of samples:", len(sample_metadata))

            ranked_scores = optimise_candidates(
                sample_metadata=sample_metadata,
                dem_path=dem_path,
                max_distance_m=max_distance,
                weights=OptimiserWeights(
                    ndvi=0.40,
                    visibility_strength=0.40,
                    unseenness=0.00,
                    obstacle_penalty=0.20,
                ),
                download_images=False,
            )

            results_df = scores_to_dataframe(ranked_scores)

            print("\n=== OPTIMISER RESULTS ===")
            print("Total points received:", len(sample_metadata))
            print("Total ranked points:", len(ranked_scores))

            columns_to_print = [
                "index",
                "lon",
                "lat",
                "heading_deg",
                "mean_ndvi",
                "ndvi_score",
                "mean_visibility_count",
                "visibility_score",
                "unseenness_score",
                "occlusion_fraction",
                "final_score",
            ]

            existing_columns = [
                column for column in columns_to_print
                if column in results_df.columns
            ]

            print(results_df[existing_columns].to_string(index=False))

            best_candidate = ranked_scores[0]

            print("\n=== BEST CANDIDATE ===")
            print(f"Index: {best_candidate.index}")
            print(f"Longitude: {best_candidate.lon}")
            print(f"Latitude: {best_candidate.lat}")
            print(f"Final score: {best_candidate.final_score}")
            print(f"Mean NDVI: {best_candidate.mean_ndvi}")
            print(f"Mean visibility count: {best_candidate.mean_visibility_count}")
            print(f"Obstacle fraction: {best_candidate.occlusion_fraction}")

            right_sidebar.canvas.draw_idle()

        except ValueError as e:
            print("Input error:", e)
            show_error(str(e))
        except Exception as e:
            print("Optimiser error:", e)
            show_error(str(e))

    global tif_path 
    root = tk.Tk() #create the GUI window.
    root.title("VISTA") #title the window.
    root.geometry("1350x760") #default window size.
    root.resizable(True, True) #allow user to resize window.

    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1) #define region for file bar.

    left_container = ttk.Frame(root) #outer container for scrollable left panel.
    left_container.grid(row=1, column=0, sticky="nsew") #place container in left side of main window.

    left_container.rowconfigure(0, weight=1)
    left_container.columnconfigure(0, weight=1)

    left_canvas = tk.Canvas(left_container, highlightthickness=0, width=620) #canvas that will scroll.
    left_canvas.grid(row=0, column=0, sticky="nsew")

    left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview) #vertical scrollbar.
    left_scrollbar.grid(row=0, column=1, sticky="ns")

    left_canvas.configure(yscrollcommand=left_scrollbar.set)

    left_panel = ttk.Frame(left_canvas, padding=12) #actual content frame that holds all widgets.
    left_panel_window = left_canvas.create_window((0, 0), window=left_panel, anchor="nw") #place frame inside canvas.

    def update_left_scrollregion(event=None):
        left_canvas.configure(scrollregion=left_canvas.bbox("all")) #tell canvas how tall the scrollable area is.

    def resize_left_panel_width(event):
        left_canvas.itemconfigure(left_panel_window, width=event.width) #keep inner frame same width as canvas.

    left_panel.bind("<Configure>", update_left_scrollregion)
    left_canvas.bind("<Configure>", resize_left_panel_width)

    def on_mousewheel_windows(event):
        left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units") #mouse wheel scroll on Windows.

    def on_mousewheel_linux_up(event):
        left_canvas.yview_scroll(-1, "units") #Linux scroll up.

    def on_mousewheel_linux_down(event):
        left_canvas.yview_scroll(1, "units") #Linux scroll down.

    def bind_mousewheel(event=None):
        left_canvas.bind_all("<MouseWheel>", on_mousewheel_windows)
        left_canvas.bind_all("<Button-4>", on_mousewheel_linux_up)
        left_canvas.bind_all("<Button-5>", on_mousewheel_linux_down)

    def unbind_mousewheel(event=None):
        left_canvas.unbind_all("<MouseWheel>")
        left_canvas.unbind_all("<Button-4>")
        left_canvas.unbind_all("<Button-5>")

    left_canvas.bind("<Enter>", bind_mousewheel)
    left_canvas.bind("<Leave>", unbind_mousewheel)

    max_distance_var = tk.StringVar(value="500.0")
    dem_source_var = tk.StringVar(value="local")

    selected_metadata_var, load_metadata_file = metadata_handler() #create button to load in a CSV by defining variable and function with does so.
    right_sidebar = RightSideBar(root) #create right hand pannel.
    right_sidebar.grid(row=1, column=1, sticky="nsew") #define region for right panel.

    lon_var = tk.StringVar()
    lat_var = tk.StringVar()
    height_var = tk.StringVar()
    heading_var = tk.StringVar()

    tk.Label(left_panel, text="Samples").grid(row=1, column=0, padx=(12, 4), pady=(8, 4), sticky="w")

    tree_frame = ttk.Frame(left_panel)
    tree_frame.grid(row=2, column=0, columnspan=9, sticky="nsew", padx=12, pady=(0, 12))

    tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
    tree_scrollbar.pack(side="right", fill="y")

    sample_tree = ttk.Treeview(
        tree_frame,
        columns=("sample", "lon", "lat", "observer_height", "heading_deg"),
        show="headings",
        height=16,
        yscrollcommand=tree_scrollbar.set,
        selectmode="browse",
    )
    sample_tree.pack(side="left", fill="both", expand=True)

    tree_scrollbar.config(command=sample_tree.yview)

    sample_tree.heading("sample", text="Sample")
    sample_tree.heading("lon", text="Longitude")
    sample_tree.heading("lat", text="Latitude")
    sample_tree.heading("observer_height", text="Observer height (m)")
    sample_tree.heading("heading_deg", text="Heading (deg)")

    sample_tree.column("sample", width=70, anchor="center")
    sample_tree.column("lon", width=140, anchor="center")
    sample_tree.column("lat", width=140, anchor="center")
    sample_tree.column("observer_height", width=140, anchor="center")
    sample_tree.column("heading_deg", width=120, anchor="center")

    def on_tree_select(event=None):
        selected = sample_tree.selection()
        if not selected:
            return

        item_id = selected[0]
        values = sample_tree.item(item_id, "values")

        lon_var.set(values[1])
        lat_var.set(values[2])
        height_var.set(values[3])
        heading_var.set(values[4])

    sample_tree.bind("<<TreeviewSelect>>", on_tree_select)

    def build_sample_from_editor():
        max_observer_height = 10000

        lon_text = lon_var.get().strip()
        lat_text = lat_var.get().strip()
        observer_height_text = height_var.get().strip()
        heading_text = heading_var.get().strip()

        if not all([lon_text, lat_text, observer_height_text, heading_text]):
            error_label.config(text="Please fill in longitude, latitude, observer height, and heading.")

        try:
            lon = float(lon_text)
            lat = float(lat_text)
            observer_height = float(observer_height_text)
            heading_deg = float(heading_text)
        except ValueError:
            error_label.config(text="Longitude, latitude, observer height, and heading must all be numbers.")

        if not (-180 <= lon <= 180):
            error_label.config(text="Longitude must be between -180 and 180.")

        if not (-90 <= lat <= 90):
            error_label.config(text="Latitude must be between -90 and 90.")

        if observer_height <= 0:
            error_label.config(text="Observer height must be greater than 0 metres.")

        if observer_height > max_observer_height:
            error_label.config(text=f"Observer height must not exceed {max_observer_height} metres.")

        if not (0 <= heading_deg < 360):
            error_label.config(text="Heading must be between 0 and 360 degrees.")

        return {
            "lon": lon,
            "lat": lat,
            "observer_height": observer_height,
            "heading_deg": heading_deg,
        }

    def renumber_tree_rows():
        for i, item_id in enumerate(sample_tree.get_children(), start=1):
            values = list(sample_tree.item(item_id, "values"))
            values[0] = i
            sample_tree.item(item_id, values=values)

    def add_sample_row():
        sample = build_sample_from_editor()

        next_number = len(sample_tree.get_children()) + 1
        item_id = sample_tree.insert(
            "",
            "end",
            values=(
                next_number,
                sample["lon"],
                sample["lat"],
                sample["observer_height"],
                sample["heading_deg"],
            ),
        )

        sample_tree.selection_set(item_id)
        sample_tree.focus(item_id)
        on_tree_select()
        error_label.config(text="")
        update_left_scrollregion()

    def update_selected_row():
        selected = sample_tree.selection()
        if not selected:
            error_label.config(text="Please select a sample row to update.")
            return

        sample = build_sample_from_editor()
        item_id = selected[0]
        old_values = list(sample_tree.item(item_id, "values"))

        sample_tree.item(
            item_id,
            values=(
                old_values[0],
                sample["lon"],
                sample["lat"],
                sample["observer_height"],
                sample["heading_deg"],
            ),
        )

        error_label.config(text="")

    def delete_selected_row():
        selected = sample_tree.selection()
        if not selected:
            error_label.config(text="Please select a sample row to delete.")
            return

        sample_tree.delete(selected[0])
        renumber_tree_rows()
        error_label.config(text="")

        children = sample_tree.get_children()
        if children:
            sample_tree.selection_set(children[0])
            sample_tree.focus(children[0])
            on_tree_select()
        else:
            lon_var.set("")
            lat_var.set("")
            height_var.set("")
            heading_var.set("")

        update_left_scrollregion()

    editor_frame = ttk.LabelFrame(left_panel, text="Selected sample")
    editor_frame.grid(row=3, column=0, columnspan=9, sticky="ew", padx=12, pady=(0, 12))

    ttk.Label(editor_frame, text="Longitude:").grid(row=0, column=0, padx=(10, 4), pady=8, sticky="w")
    ttk.Entry(editor_frame, textvariable=lon_var, width=16).grid(row=0, column=1, padx=(0, 4), pady=8, sticky="w")

    lon_help = tk.Label(editor_frame, text="?", fg="blue", cursor="question_arrow")
    lon_help.grid(row=0, column=2, padx=(0, 12), pady=8, sticky="w")

    ttk.Label(editor_frame, text="Latitude:").grid(row=0, column=3, padx=(10, 4), pady=8, sticky="w")
    ttk.Entry(editor_frame, textvariable=lat_var, width=16).grid(row=0, column=4, padx=(0, 4), pady=8, sticky="w")

    lat_help = tk.Label(editor_frame, text="?", fg="blue", cursor="question_arrow")
    lat_help.grid(row=0, column=5, padx=(0, 12), pady=8, sticky="w")

    ttk.Label(editor_frame, text="Observer height (m):").grid(row=1, column=0, padx=(10, 4), pady=8, sticky="w")
    ttk.Entry(editor_frame, textvariable=height_var, width=16).grid(row=1, column=1, padx=(0, 12), pady=8, sticky="w")

    ttk.Label(editor_frame, text="Heading (deg):").grid(row=1, column=3, padx=(10, 4), pady=8, sticky="w")
    ttk.Entry(editor_frame, textvariable=heading_var, width=16).grid(row=1, column=4, padx=(0, 12), pady=8, sticky="w")

    LeftSideBar(
        lon_help,
        "Enter longitude in EPSG:4326 (WGS84).\nExample: -1.327600"
    )

    LeftSideBar(
        lat_help,
        "Enter latitude in EPSG:4326 (WGS84).\nExample: 50.730251"
    )

    button_frame = tk.Frame(left_panel)
    button_frame.grid(row=4, column=0, columnspan=9, pady=(0, 10), sticky="w", padx=12)

    add_row_button = tk.Button(button_frame, text="Add sample", command=add_sample_row)
    add_row_button.pack(side="left")

    update_row_button = tk.Button(button_frame, text="Update selected", command=update_selected_row)
    update_row_button.pack(side="left", padx=(8, 0))

    delete_row_button = tk.Button(button_frame, text="Delete selected", command=delete_selected_row)
    delete_row_button.pack(side="left", padx=(8, 0))

    submit_button = tk.Button(left_panel, text="Submit", command=submit)
    submit_button.grid(row=5, column=0, columnspan=9, pady=(10, 10))

    error_label = tk.Label(left_panel, text="", fg="red")
    error_label.grid(row=6, column=0, columnspan=9, pady=(0, 10))

    # start with one blank row in the editor, but no tree rows yet
    lon_var.set("")
    lat_var.set("")
    height_var.set("")
    heading_var.set("")


    update_left_scrollregion()

    file_handler()

    if tif_path:
        right_sidebar.load_dem(tif_path) #load initial DEM preview.

    #start Tkinter event loop so it "listens" for user input.
    root.mainloop()