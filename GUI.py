from vista import run_program

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
from API_caller import download_dem_for_samples


tif_path = None
metadata_csv_path = None
loaded_sample_metadata = []


def start_gui(run_program):
    def set_coordinate_entries(lon, lat):
        row_index = selected_row_var.get() - 1
        if not (0 <= row_index < len(sample_entries)):
            return

        lon_entry, lat_entry, _, _ = sample_entries[row_index]

        lon_entry.delete(0, tk.END)
        lon_entry.insert(0, f"{lon:.6f}")

        lat_entry.delete(0, tk.END)
        lat_entry.insert(0, f"{lat:.6f}")

    class RightSideBar(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent, padding=8)

            self.dem = None
            self.dem_transform = None
            self.dem_crs = None
            self.dem_path = None
            self.count_overlay = None
            self.observer_points_xy = []

            self.point_selected_callback = None
            self.clicked_points = []
            self.tip = None

            self.rowconfigure(1, weight=1)
            self.columnconfigure(0, weight=1)

            self.title_label = ttk.Label(self, text="DEM PREVIEW", font=("Segoe UI", 12, "bold"))
            self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

            self.fig = Figure(figsize=(11, 5.8), dpi=100)
            self.ax_count = self.fig.add_subplot(111)
            self.ax_count.set_title("No DEM loaded")
            self.ax_count.set_xticks([])
            self.ax_count.set_yticks([])

            self.canvas = FigureCanvasTkAgg(self.fig, master=self)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.grid(row=1, column=0, sticky="nsew")

            self.toolbar_frame = ttk.Frame(self)
            self.toolbar_frame.grid(row=2, column=0, sticky="ew")
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.pack(side="left")

            self.show_overlay = tk.BooleanVar(value=True)
            self.count_overlay_im = None
            self.count_overlay_cbar = None

            self.view_xlim = None
            self.view_ylim = None
            self.view_extent = None

            menubar = tk.Menu(parent)
            parent.config(menu=menubar)

            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Load metadata CSV", command=load_metadata_file)

            view_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="View", menu=view_menu)
            view_menu.add_checkbutton(
                label="Show overlay",
                variable=self.show_overlay,
                command=self.toggle_overlay,
            )

            self.canvas.mpl_connect("button_press_event", self.on_click)

        def on_click(self, event):
            if self.dem is None:
                return
            if event.inaxes != self.ax_count:
                return
            if event.xdata is None or event.ydata is None:
                return
            if self.toolbar.mode != "":
                return

            x = event.xdata
            y = event.ydata
            self.clicked_points.append((x, y))

            if self.dem_crs is not None and self.point_selected_callback is not None:
                transformer = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(x, y)
                self.point_selected_callback(lon, lat)

            self._redraw()

        def load_dem(self, dem_path):
            with rasterio.open(dem_path) as src:
                self.dem = src.read(1, masked=True)
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_path = dem_path

            self.count_overlay = None
            self.observer_points_xy = []
            self._redraw()

        def set_results(self, count_overlay, observer_points=None, view_extent=None):
            if self.dem is None:
                raise ValueError("Load a DEM before setting an overlay.")

            self.count_overlay = count_overlay
            self.observer_points_xy = observer_points if observer_points is not None else []
            self.view_extent = view_extent
            self.view_xlim = None
            self.view_ylim = None
            self._redraw()

        def toggle_overlay(self):
            show_state = self.show_overlay.get()

            if self.count_overlay_im is not None:
                self.count_overlay_im.set_visible(show_state)

            if self.count_overlay_cbar is not None:
                self.count_overlay_cbar.ax.set_visible(show_state)

            self.canvas.draw_idle()

        def clear_overlay(self):
            self.count_overlay = None
            self.observer_points_xy = []
            self._redraw()

        def _remove_colourbars(self):
            if self.count_overlay_cbar is not None:
                self.count_overlay_cbar.remove()
                self.count_overlay_cbar = None

        def _redraw(self):
            self.ax_count.clear()
            self._remove_colourbars()

            if self.dem is None:
                self.ax_count.set_title("No DEM loaded")
                self.ax_count.set_xticks([])
                self.ax_count.set_yticks([])
                self.canvas.draw_idle()
                return

            show(
                self.dem,
                transform=self.dem_transform,
                ax=self.ax_count,
                cmap="terrain",
            )

            self.ax_count.set_title("Frequency count heatmap")

            if self.count_overlay is not None and self.show_overlay.get():
                count_overlay = np.ma.masked_where(self.count_overlay == 0, self.count_overlay)
                count_vmax = int(np.max(self.count_overlay)) if np.any(self.count_overlay > 0) else 1

                cmap = cm.get_cmap("plasma").copy()
                cmap.set_bad(color="lightgrey")  # zero visibility

                norm = Normalize(vmin=1, vmax=max(1, count_vmax))

                show(
                    count_overlay,
                    transform=self.dem_transform,
                    ax=self.ax_count,
                    cmap=cmap,
                    norm=norm,
                    alpha=1.0,
                    zorder=20
                )

                self.count_overlay_im = self.ax_count.images[-1]
                self.count_overlay_cbar = self.fig.colorbar(
                    self.count_overlay_im,
                    ax=self.ax_count,
                    label="Number of sightings"
                )
            else:
                self.count_overlay_im = None

            if self.observer_points_xy:
                xs = [x for x, _ in self.observer_points_xy]
                ys = [y for _, y in self.observer_points_xy]
                import matplotlib.patches as mpatches
                no_vis_patch = mpatches.Patch(color="lightgrey", label="No visibility")
                self.ax_count.legend(handles=[no_vis_patch], loc="lower right")
                

            self.ax_count.set_xlabel("Easting (m)")
            self.ax_count.set_ylabel("Northing (m)")

            if self.view_xlim is not None and self.view_ylim is not None:
                self.ax_count.set_xlim(self.view_xlim)
                self.ax_count.set_ylim(self.view_ylim)
            elif self.view_extent is not None:
                left, right, bottom, top = self.view_extent
                self.ax_count.set_xlim(left, right)
                self.ax_count.set_ylim(bottom, top)

            self.fig.tight_layout()
            self.canvas.draw_idle()

    class LeftSideBar:
        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip = None

            self.widget.bind("<Enter>", self.show_tip)
            self.widget.bind("<Leave>", self.hide_tip)

        def show_tip(self, event=None):
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + 20

            self.tip = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")

            label = tk.Label(
                tw,
                text=self.text,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padx=6,
                pady=4,
            )
            label.pack()

        def hide_tip(self, event=None):
            if self.tip is not None:
                self.tip.destroy()
                self.tip = None

    def populate_sample_entries(samples):
        for lon_entry, lat_entry, height_entry, heading_entry in sample_entries:
            lon_entry.delete(0, tk.END)
            lat_entry.delete(0, tk.END)
            height_entry.delete(0, tk.END)
            heading_entry.delete(0, tk.END)

        for i, sample in enumerate(samples[:10]):
            lon_entry, lat_entry, height_entry, heading_entry = sample_entries[i]
            lon_entry.insert(0, str(sample["lon"]))
            lat_entry.insert(0, str(sample["lat"]))
            height_entry.insert(0, str(sample["observer_height"]))
            heading_entry.insert(0, str(sample["heading_deg"]))

    def load_metadata_csv(file_path):
        samples = []

        with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)

            required_columns = {"lon", "lat", "observer_height", "heading_deg"}
            if reader.fieldnames is None:
                raise ValueError("The CSV file has no header row.")

            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing))}")

            for row_index, row in enumerate(reader, start=2):
                try:
                    lon = float(row["lon"])
                    lat = float(row["lat"])
                    observer_height = float(row["observer_height"])
                    heading_deg = float(row["heading_deg"])
                except ValueError:
                    raise ValueError(f"Invalid numeric value in CSV row {row_index}.")

                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude out of range in CSV row {row_index}.")
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude out of range in CSV row {row_index}.")
                if observer_height <= 0:
                    raise ValueError(f"Observer height must be greater than 0 in CSV row {row_index}.")
                if not (0 <= heading_deg < 360):
                    raise ValueError(f"Heading must be between 0 and 360 degrees in CSV row {row_index}.")

                samples.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "observer_height": observer_height,
                        "heading_deg": heading_deg,
                    }
                )

        if len(samples) != 10:
            raise ValueError(f"CSV must contain exactly 10 samples for this proof of concept. Found {len(samples)}.")

        return samples

    def metadata_handler():
        selected_metadata_var = tk.StringVar(value=metadata_csv_path if metadata_csv_path else "No metadata CSV selected")

        def load_metadata_file():
            global metadata_csv_path
            global loaded_sample_metadata

            file_path = filedialog.askopenfilename(
                parent=root,
                title="Open metadata CSV for VISTA",
                initialdir=".",
                filetypes=[("CSV files", "*.csv")],
            )

            if not file_path:
                return

            try:
                samples = load_metadata_csv(file_path)
                metadata_csv_path = file_path
                loaded_sample_metadata = samples
                selected_metadata_var.set(file_path)
                populate_sample_entries(samples)
                error_label.config(text="")
            except Exception as e:
                messagebox.showerror("Load error", f"Could not load metadata CSV:\n{e}")

        return selected_metadata_var, load_metadata_file

    def validate_inputs():
        if loaded_sample_metadata:
            return loaded_sample_metadata

        max_observer_height = 10000
        samples = []

        for idx, (lon_entry, lat_entry, height_entry, heading_entry) in enumerate(sample_entries, start=1):
            lon_text = lon_entry.get().strip()
            lat_text = lat_entry.get().strip()
            observer_height_text = height_entry.get().strip()
            heading_text = heading_entry.get().strip()

            if not lon_text or not lat_text or not observer_height_text or not heading_text:
                raise ValueError(f"Please fill in longitude, latitude, observer height, and heading for sample {idx}.")

            try:
                lon = float(lon_text)
                lat = float(lat_text)
                observer_height = float(observer_height_text)
                heading_deg = float(heading_text)
            except ValueError:
                raise ValueError(f"Longitude, latitude, observer height, and heading must all be numbers for sample {idx}.")

            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude must be between -180 and 180 for sample {idx}.")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90 for sample {idx}.")
            if observer_height <= 0:
                raise ValueError(f"Observer height must be greater than 0 metres for sample {idx}.")
            if observer_height > max_observer_height:
                raise ValueError(f"Observer height must not exceed {max_observer_height} metres for sample {idx}.")
            if not (0 <= heading_deg < 360):
                raise ValueError(f"Heading must be between 0 and 360 degrees for sample {idx}.")

            samples.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "observer_height": observer_height,
                    "heading_deg": heading_deg,
                }
            )

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
        error_label.config(text=message)

    def file_handler():
        selected_file_var = tk.StringVar(value=tif_path if tif_path else "No file selected")

        def validate_file(file_path):
            ext = Path(file_path).suffix.lower()
            if ext not in [".tif", ".tiff"]:
                raise ValueError(f"Unsupported file type: {ext}")

        def load_file():
            global tif_path
            file_path = filedialog.askopenfilename(
                parent=root,
                title="Open file for VISTA",
                initialdir=".",
                filetypes=[("GeoTIFF files", "*.tif *.tiff")],
            )

            if not file_path:
                return

            try:
                validate_file(file_path)
                tif_path = file_path
                selected_file_var.set(file_path)
                right_sidebar.load_dem(tif_path)
                error_label.config(text="")
            except Exception as e:
                messagebox.showerror("Load error", f"Could not load file:\n{e}")

        top_bar = ttk.Frame(root, padding=8)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        file_button = ttk.Button(top_bar, text="DEM", command=load_file)
        file_button.pack(side="left")

        file_label = ttk.Label(top_bar, textvariable=selected_file_var)
        file_label.pack(side="left", padx=10)

        metadata_button = ttk.Button(top_bar, text="CSV", command=load_metadata_file)
        metadata_button.pack(side="left", padx=(20, 0))

        metadata_label = ttk.Label(top_bar, textvariable=selected_metadata_var)
        metadata_label.pack(side="left", padx=10)

        max_distance_label = ttk.Label(top_bar, text="Max distance (m):")
        max_distance_label.pack(side="left", padx=(20, 4))

        max_distance_entry = ttk.Entry(top_bar, width=10, textvariable=max_distance_var)
        max_distance_entry.pack(side="left")

        dem_mode_label = ttk.Label(top_bar, text="DEM source:")
        dem_mode_label.pack(side="left", padx=(20, 4))

        local_radio = ttk.Radiobutton(
            top_bar,
            text="Use downloaded DEM",
            variable=dem_source_var,
            value="local",
        )
        local_radio.pack(side="left")

        download_radio = ttk.Radiobutton(
            top_bar,
            text="Download DEM around locations",
            variable=dem_source_var,
            value="download",
        )
        download_radio.pack(side="left", padx=(10, 0))

    preview_window = None

    def open_visibility_window(image_path):
        nonlocal preview_window

        path = Path(image_path)
        if not path.exists():
            messagebox.showwarning(
                "Preview not found",
                f"Could not find:\n{path.resolve()}",
            )
            return

        if preview_window is not None and preview_window.winfo_exists():
            preview_window.destroy()

        preview_window = tk.Toplevel(root)
        preview_window.title("Visibility frequency")
        preview_window.geometry("1100x850")

        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)

        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis("off")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=preview_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        preview_window._fig = fig
        preview_window._canvas = canvas

    def submit():
        global tif_path
        error_label.config(text="")

        try:
            sample_metadata = validate_inputs()
            max_distance = validate_max_distance()
            dem_path = get_dem_path(sample_metadata, max_distance)

            right_sidebar.load_dem(dem_path)

            result = run_program(sample_metadata, dem_path, max_distance, show_reference=False)

            right_sidebar.set_results(
                result["count_overlay"],
                observer_points=result["observer_points_xy"],
                view_extent=result["view_extent"],
            )

            right_sidebar.toggle_overlay()
            right_sidebar.canvas.draw_idle()

            preview_png_path = result.get(
                    "preview_png_path",
                    "visibility_frequency_cropped.png",
                )
            open_visibility_window(preview_png_path)

        except ValueError as e:
            show_error(str(e))
        except Exception as e:
            show_error(str(e))

    global tif_path
    root = tk.Tk()
    root.title("VISTA")
    root.geometry("1350x760")
    root.resizable(True, True)

    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1)

    left_panel = ttk.Frame(root, padding=12)
    left_panel.grid(row=1, column=0, sticky="ns")

    max_distance_var = tk.StringVar(value="500.0")
    dem_source_var = tk.StringVar(value="local") # "local" = use the existing DEM file picker // "download" = fetch a DEM from OpenTopography

    selected_metadata_var, load_metadata_file = metadata_handler()
    right_sidebar = RightSideBar(root)
    right_sidebar.grid(row=1, column=1, sticky="nsew")

    selected_row_var = tk.IntVar(value=1)
    sample_entries = []

    tk.Label(left_panel, text="Sample").grid(row=1, column=0, padx=(12, 4), pady=8, sticky="w")
    tk.Label(left_panel, text="Longitude (EPSG:4326):").grid(row=1, column=1, padx=(12, 4), pady=8, sticky="w")
    lon_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow")
    lon_help.grid(row=1, column=2, padx=6, pady=8, sticky="w")

    tk.Label(left_panel, text="Latitude (EPSG:4326):").grid(row=1, column=3, padx=(12, 4), pady=8, sticky="w")
    lat_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow")
    lat_help.grid(row=1, column=4, padx=6, pady=8, sticky="w")

    tk.Label(left_panel, text="Observer height (m):").grid(row=1, column=5, padx=(12, 4), pady=8, sticky="w")
    height_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow")
    height_help.grid(row=1, column=6, padx=6, pady=8, sticky="w")

    tk.Label(left_panel, text="Heading (deg):").grid(row=1, column=7, padx=(12, 4), pady=8, sticky="w")
    heading_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow")
    heading_help.grid(row=1, column=8, padx=6, pady=8, sticky="w")

    for i in range(10):
        tk.Label(left_panel, text=f"{i + 1}.").grid(row=i + 2, column=0, padx=(12, 4), pady=4, sticky="w")

        lon_entry = tk.Entry(left_panel, width=16)
        lon_entry.grid(row=i + 2, column=1, pady=4, sticky="w")

        lat_entry = tk.Entry(left_panel, width=16)
        lat_entry.grid(row=i + 2, column=3, pady=4, sticky="w")

        height_entry = tk.Entry(left_panel, width=12)
        height_entry.grid(row=i + 2, column=5, pady=4, sticky="w")

        heading_entry = tk.Entry(left_panel, width=12)
        heading_entry.grid(row=i + 2, column=7, pady=4, sticky="w")

        sample_entries.append((lon_entry, lat_entry, height_entry, heading_entry))

    right_sidebar.point_selected_callback = set_coordinate_entries

    submit_button = tk.Button(left_panel, text="Submit", command=submit)
    submit_button.grid(row=12, column=0, columnspan=9, pady=(18, 10))

    error_label = tk.Label(left_panel, text="", fg="red")
    error_label.grid(row=13, column=0, columnspan=9, pady=(0, 10))

    file_handler()

    LeftSideBar(lon_help, "Enter the coordinate's longitude in EPSG:4326.\nExample: -1.3276")
    LeftSideBar(lat_help, "Enter the coordinate's latitude in EPSG:4326.\nExample: 50.730251")
    LeftSideBar(height_help, "Enter observer height above the ground in metres.\nExample: 1.5")
    LeftSideBar(
        heading_help,
        "Enter the viewing direction as a compass bearing in degrees.\n0 = north, 90 = east, 180 = south, 270 = west",
    )

    if tif_path:
        right_sidebar.load_dem(tif_path)

    root.mainloop()