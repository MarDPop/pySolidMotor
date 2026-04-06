from motor import *

import sys
import traceback
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


DEFAULTS = {
    "pressure_ambient": 101325.0,
    "temperature_ambient": 293.15,
    "gamma": 1.132,
    "mw": 0.043,
    "cfl": 0.8,
    "dt_record": 0.01,
    "max_steps": 20000000000,
    "case_length": 0.25,
    "case_radius": 0.05,
    "throat_radius": 0.010,
    "chamber_radius": 0.05,
    "exit_radius": 0.0183,
    "discharge_coefficient": 0.98,
    "nozzle_efficiency": 0.95,
    "length_fraction": 0.85,
    "fuel_density": 1850.0,
    "fuel_heating_value": 2.23e6,
    "burn_a": 0.005,
    "burn_n": 0.4,
    "burn_a0": 0.0,
    "use_burn_rate_table": 0,
    "burn_rate_pressures": "101325, 700000, 1.448e+6, 2.413e+6, 3.103e+6, 3.792e+6, 4.482e+6, 5.861e+6",
    "burn_rate_values": "0.002, 0.0071, 0.0072, 0.0077, 0.0081, 0.0091, 0.011, 0.0127",
    "grain_length": 0.20,
    "outer_radius": 0.05,
    "inner_radius": 0.015,
    "exposed_caps": 2,
}

GEOMETRY_FIELDS = (
    ("Case Length (m)", "case_length"),
    ("Case Radius (m)", "case_radius"),
    ("Throat Radius (m)", "throat_radius"),
    ("Nozzle Chamber Radius (m)", "chamber_radius"),
    ("Exit Radius (m)", "exit_radius"),
    ("Discharge Coeff", "discharge_coefficient"),
    ("Nozzle Efficiency", "nozzle_efficiency"),
    ("Rao Length Fraction", "length_fraction"),
    ("Grain Length (m)", "grain_length"),
    ("Outer Radius (m)", "outer_radius"),
    ("Inner Radius (m)", "inner_radius"),
    ("Exposed Caps", "exposed_caps"),
)

SIM_FIELDS = (
    ("Ambient Pressure (Pa)", "pressure_ambient"),
    ("Ambient Temp (K)", "temperature_ambient"),
    ("Gamma", "gamma"),
    ("Molecular Weight (kg/mol)", "mw"),
    ("CFL", "cfl"),
    ("Record Step (s)", "dt_record"),
    ("Max Steps", "max_steps"),
)

FUEL_FIELDS = (
    ("Fuel Density (kg/m^3)", "fuel_density"),
    ("Heating Value (J/kg)", "fuel_heating_value"),
    ("Burn a", "burn_a"),
    ("Burn n", "burn_n"),
    ("Burn a0", "burn_a0"),
)


def build_motor(params: dict[str, float | np.ndarray]) -> Motor:
    shape_x, shape_r = Nozzle.generate_rao(
        float(params["throat_radius"]),
        (float(params["exit_radius"]) / float(params["throat_radius"])) ** 2,
        float(params["length_fraction"]),
    )
    nozzle = Nozzle(
        float(params["throat_radius"]),
        float(params["chamber_radius"]),
        float(params["exit_radius"]),
        float(params["discharge_coefficient"]),
        float(params["nozzle_efficiency"]),
        shape_x,
        shape_r,
    )

    case = Case()
    case.length = float(params["case_length"])
    case.radius = float(params["case_radius"])

    if bool(params.get("use_burn_rate_table", 0)):
        fuel = Fuel_LUT(
            float(params["fuel_density"]),
            float(params["fuel_heating_value"]),
            np.asarray(params["burn_rate_pressures"], dtype=float),
            np.asarray(params["burn_rate_values"], dtype=float),
        )
    else:
        fuel = Fuel_StRoberts(
            float(params["fuel_density"]),
            float(params["fuel_heating_value"]),
            float(params["burn_a"]),
            float(params["burn_n"]),
            float(params["burn_a0"]),
        )

    grain = GrainSegment(
        fuel,
        HollowCylinderCrossSection(float(params["outer_radius"]), float(params["inner_radius"])),
        int(float(params["exposed_caps"])),
    )
    grain.length = float(params["grain_length"])

    motor = Motor(nozzle, case)
    motor.segments.append(grain)
    return motor


class MotorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Motor Performance Simulator")
        self.inputs: dict[str, tk.StringVar] = {}
        self.entry_widgets: dict[str, ttk.Entry] = {}
        self.manual_burn_rate_var = tk.BooleanVar(value=bool(DEFAULTS["use_burn_rate_table"]))
        self._geometry_redraw_after_id: str | None = None
        self.latest_velocity_profile: tuple[np.ndarray, np.ndarray] | None = None

        container = ttk.Frame(root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        controls_outer = ttk.Frame(container)
        controls_outer.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        controls_outer.columnconfigure(0, weight=1)
        controls_outer.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(container)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(controls_outer, width=360, highlightthickness=0)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(
            controls_outer, orient="vertical", command=self.controls_canvas.yview
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        self.controls_frame = ttk.Frame(self.controls_canvas)
        self.controls_window = self.controls_canvas.create_window(
            (0, 0), window=self.controls_frame, anchor="nw"
        )
        self.controls_frame.bind("<Configure>", self._sync_scroll_region)
        self.controls_canvas.bind("<Configure>", self._resize_scroll_window)
        self.controls_canvas.bind("<Enter>", self._bind_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_mousewheel)

        geometry_box = ttk.LabelFrame(self.controls_frame, text="Geometry Parameters", padding=10)
        geometry_box.grid(row=0, column=0, sticky="ew")
        sim_box = ttk.LabelFrame(self.controls_frame, text="Simulation Parameters", padding=10)
        sim_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        fuel_box = ttk.LabelFrame(self.controls_frame, text="Fuel Parameters", padding=10)
        fuel_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        actions = ttk.Frame(self.controls_frame)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.controls_frame.columnconfigure(0, weight=1)

        self._build_fields(geometry_box, GEOMETRY_FIELDS)
        self._build_fields(sim_box, SIM_FIELDS)
        self._build_fields(fuel_box, FUEL_FIELDS)
        self._build_burn_rate_table_controls(fuel_box)

        ttk.Button(actions, text="Run Simulation", command=self.run_simulation).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(
            actions,
            text="Show Nozzle Velocity Profile",
            command=self.show_velocity_profile,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.status_var = tk.StringVar(value="Edit geometry to preview the motor, then run the simulation.")
        ttk.Label(actions, textvariable=self.status_var, wraplength=280).grid(
            row=2, column=0, sticky="ew", pady=(0, 8)
        )

        self.summary_var = tk.StringVar(value="")
        ttk.Label(actions, textvariable=self.summary_var, justify="left", wraplength=280).grid(
            row=3, column=0, sticky="ew"
        )

        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.ax_geometry = self.figure.add_subplot(311)
        self.ax_thrust = self.figure.add_subplot(312)
        self.ax_mass = self.figure.add_subplot(313)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._reset_plots()
        self._update_burn_rate_mode()
        self._refresh_geometry_preview()

    def _build_fields(self, parent: ttk.LabelFrame, fields: tuple[tuple[str, str], ...]) -> None:
        for row, (label, key) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=str(DEFAULTS[key]))
            self.inputs[key] = var
            var.trace_add("write", self._schedule_geometry_preview)
            entry = ttk.Entry(parent, textvariable=var, width=16)
            entry.grid(row=row, column=1, sticky="ew", pady=1, padx=(8, 0))
            self.entry_widgets[key] = entry
        parent.columnconfigure(1, weight=1)

    def _build_burn_rate_table_controls(self, parent: ttk.LabelFrame) -> None:
        start_row = len(FUEL_FIELDS)
        ttk.Checkbutton(
            parent,
            text="Use manual pressure / burn-rate arrays",
            variable=self.manual_burn_rate_var,
            command=self._update_burn_rate_mode,
        ).grid(row=start_row, column=0, columnspan=2, sticky="w", pady=(8, 4))

        self.manual_table_frame = ttk.Frame(parent)
        self.manual_table_frame.grid(row=start_row + 1, column=0, columnspan=2, sticky="ew")
        self.manual_table_frame.columnconfigure(0, weight=1)

        ttk.Label(self.manual_table_frame, text="Pressures (Pa)").grid(row=0, column=0, sticky="w")
        self.pressures_text = tk.Text(self.manual_table_frame, height=4, width=28)
        self.pressures_text.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.pressures_text.insert("1.0", str(DEFAULTS["burn_rate_pressures"]))
        self.pressures_text.bind("<<Modified>>", self._on_table_text_modified)
        self.pressures_text.edit_modified(False)

        ttk.Label(self.manual_table_frame, text="Burn rates (m/s)").grid(row=2, column=0, sticky="w")
        self.burn_rates_text = tk.Text(self.manual_table_frame, height=4, width=28)
        self.burn_rates_text.grid(row=3, column=0, sticky="ew")
        self.burn_rates_text.insert("1.0", str(DEFAULTS["burn_rate_values"]))
        self.burn_rates_text.bind("<<Modified>>", self._on_table_text_modified)
        self.burn_rates_text.edit_modified(False)

    def _sync_scroll_region(self, event: tk.Event | None = None) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_scroll_window(self, event: tk.Event) -> None:
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _bind_mousewheel(self, event: tk.Event | None = None) -> None:
        self.controls_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event: tk.Event | None = None) -> None:
        self.controls_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.controls_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_table_text_modified(self, event: tk.Event) -> None:
        widget = event.widget
        if not widget.edit_modified():
            return
        self._schedule_geometry_preview()
        widget.edit_modified(False)

    def _update_burn_rate_mode(self) -> None:
        use_table = self.manual_burn_rate_var.get()
        st_roberts_state = "disabled" if use_table else "normal"
        text_state = "normal" if use_table else "disabled"

        for key in ("burn_a", "burn_n", "burn_a0"):
            self.entry_widgets[key].configure(state=st_roberts_state)

        self.pressures_text.configure(state=text_state)
        self.burn_rates_text.configure(state=text_state)
        self._schedule_geometry_preview()

    def _parse_float_list(self, raw_value: str, label: str) -> np.ndarray:
        values = [token.strip() for token in raw_value.replace("\n", ",").split(",") if token.strip()]
        if not values:
            raise ValueError(f"{label} cannot be empty.")
        try:
            return np.asarray([float(value) for value in values], dtype=float)
        except ValueError as exc:
            raise ValueError(f"{label} must contain only numbers.") from exc

    def _read_params(self) -> dict[str, float | np.ndarray]:
        params: dict[str, float | np.ndarray] = {}
        for key, var in self.inputs.items():
            params[key] = float(var.get())

        params["use_burn_rate_table"] = float(self.manual_burn_rate_var.get())
        if self.manual_burn_rate_var.get():
            params["burn_rate_pressures"] = self._parse_float_list(
                self.pressures_text.get("1.0", "end-1c"),
                "Burn rate pressures",
            )
            params["burn_rate_values"] = self._parse_float_list(
                self.burn_rates_text.get("1.0", "end-1c"),
                "Burn rate values",
            )
        else:
            params["burn_rate_pressures"] = self._parse_float_list(
                self.pressures_text.get("1.0", "end-1c"),
                "Burn rate pressures",
            )
            params["burn_rate_values"] = self._parse_float_list(
                self.burn_rates_text.get("1.0", "end-1c"),
                "Burn rate values",
            )
        return params

    def _read_geometry_params(self) -> dict[str, float | np.ndarray]:
        params: dict[str, float | np.ndarray] = {}
        for _, key in GEOMETRY_FIELDS:
            params[key] = float(self.inputs[key].get())

        params["fuel_density"] = DEFAULTS["fuel_density"]
        params["fuel_heating_value"] = DEFAULTS["fuel_heating_value"]
        params["burn_a"] = DEFAULTS["burn_a"]
        params["burn_n"] = DEFAULTS["burn_n"]
        params["burn_a0"] = DEFAULTS["burn_a0"]
        params["use_burn_rate_table"] = 0.0
        params["burn_rate_pressures"] = self._parse_float_list(
            self.pressures_text.get("1.0", "end-1c"),
            "Burn rate pressures",
        )
        params["burn_rate_values"] = self._parse_float_list(
            self.burn_rates_text.get("1.0", "end-1c"),
            "Burn rate values",
        )
        return params

    def _reset_plots(self) -> None:
        self.ax_geometry.clear()
        self.ax_thrust.clear()
        self.ax_mass.clear()
        self.ax_geometry.set_title("Motor Geometry")
        self.ax_geometry.set_xlabel("Axial Position (m)")
        self.ax_geometry.set_ylabel("Radius (m)")
        self.ax_thrust.set_title("Thrust vs Time")
        self.ax_thrust.set_ylabel("Thrust (N)")
        self.ax_mass.set_title("Fuel Mass vs Time")
        self.ax_mass.set_xlabel("Time (s)")
        self.ax_mass.set_ylabel("Mass (kg)")
        self.canvas.draw_idle()

    def _schedule_geometry_preview(self, *args) -> None:
        if self._geometry_redraw_after_id is not None:
            self.root.after_cancel(self._geometry_redraw_after_id)
        self._geometry_redraw_after_id = self.root.after(150, self._refresh_geometry_preview)

    def _refresh_geometry_preview(self) -> None:
        self._geometry_redraw_after_id = None
        try:
            params = self._read_geometry_params()
            motor = build_motor(params)
            self._plot_geometry(motor, params)
            self.figure.tight_layout(pad=2.0)
            self.canvas.draw_idle()
        except Exception:
            self.ax_geometry.clear()
            self.ax_geometry.set_title("Motor Geometry")
            self.ax_geometry.set_xlabel("Axial Position (m)")
            self.ax_geometry.set_ylabel("Radius (m)")
            self.ax_geometry.text(
                0.5,
                0.5,
                "Geometry preview unavailable\nCheck parameter values",
                ha="center",
                va="center",
                transform=self.ax_geometry.transAxes,
            )
            self.ax_geometry.grid(True, alpha=0.3)
            self.canvas.draw_idle()

    def _plot_geometry(self, motor: Motor, params: dict[str, float | np.ndarray]) -> None:
        x_nozzle, r_nozzle = motor.nozzle.full_shape()
        chamber_front_x = float(np.min(x_nozzle))
        chamber_back_x = chamber_front_x - float(params["case_length"])
        chamber_x = np.array([chamber_back_x, chamber_front_x])
        chamber_r = np.full_like(chamber_x, float(params["case_radius"]))
        grain_start = chamber_back_x
        grain_end = chamber_back_x + float(params["grain_length"])

        self.ax_geometry.clear()
        self.ax_geometry.plot(chamber_x, chamber_r, color="black", linewidth=2)
        self.ax_geometry.plot(chamber_x, -chamber_r, color="black", linewidth=2)
        self.ax_geometry.plot(
            [chamber_back_x, chamber_back_x],
            [-chamber_r[0], chamber_r[0]],
            color="black",
            linewidth=2,
        )
        self.ax_geometry.plot(x_nozzle, r_nozzle, color="tab:green", linewidth=2)
        self.ax_geometry.plot(x_nozzle, -r_nozzle, color="tab:green", linewidth=2)

        grain_outer = float(params["outer_radius"])
        grain_inner = float(params["inner_radius"])
        self.ax_geometry.fill_between(
            [grain_start, grain_end],
            [grain_outer, grain_outer],
            [grain_inner, grain_inner],
            color="tab:orange",
            alpha=0.35,
        )
        self.ax_geometry.fill_between(
            [grain_start, grain_end],
            [-grain_inner, -grain_inner],
            [-grain_outer, -grain_outer],
            color="tab:orange",
            alpha=0.35,
        )

        self.ax_geometry.axhline(0.0, color="0.5", linewidth=0.8, alpha=0.5)
        self.ax_geometry.set_title("Motor Geometry")
        self.ax_geometry.set_xlabel("Axial Position (m)")
        self.ax_geometry.set_ylabel("Radius (m)")
        self.ax_geometry.set_aspect("equal", adjustable="datalim")
        self.ax_geometry.grid(True, alpha=0.3)

    def show_velocity_profile(self) -> None:
        if self.latest_velocity_profile is None:
            self.status_var.set("Run a simulation first to view the nozzle velocity profile.")
            return

        x_centers, velocity = self.latest_velocity_profile
        popup = tk.Toplevel(self.root)
        popup.title("Nozzle Velocity Profile")
        popup.geometry("800x500")

        figure = Figure(figsize=(8, 4.5), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot(x_centers, velocity, color="tab:purple", linewidth=2)
        ax.set_title("Nozzle Velocity Profile")
        ax.set_xlabel("Axial Position (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.grid(True, alpha=0.3)
        figure.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(figure, master=popup)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw_idle()

    def run_simulation(self) -> None:
        try:
            params = self._read_params()
            motor = build_motor(params)
            self.status_var.set("Running simulation...")
            self.root.update_idletasks()

            motor.compute_constant_gamma_ideal_gas(
                float(params["pressure_ambient"]),
                float(params["temperature_ambient"]),
                float(params["gamma"]),
                float(params["mw"]),
                CFL=float(params["cfl"]),
                dt_record=float(params["dt_record"]),
                record_full_output=True,
                MAX_STEPS=int(float(params["max_steps"])),
            )

            times = np.asarray(motor.times(), dtype=float)
            thrust = np.asarray(motor.thrusts(), dtype=float)
            masses = np.asarray(motor.masses(), dtype=float)

            if times.size == 0:
                raise ValueError("Simulation produced no samples.")

            mesh = NozzleMesh(motor.nozzle)
            last_cells = motor.cells()[-1]
            rho = last_cells[0, 1:-1]
            velocity = last_cells[1, 1:-1] / rho
            self.latest_velocity_profile = (mesh.x_centers, velocity)

            self._plot_geometry(motor, params)
            self.ax_thrust.clear()
            self.ax_mass.clear()
            self.ax_thrust.plot(times, thrust, color="tab:red", linewidth=2)
            self.ax_mass.plot(times, masses, color="tab:blue", linewidth=2)
            self.ax_thrust.set_title("Thrust vs Time")
            self.ax_thrust.set_ylabel("Thrust (N)")
            self.ax_mass.set_title("Fuel Mass vs Time")
            self.ax_mass.set_xlabel("Time (s)")
            self.ax_mass.set_ylabel("Mass (kg)")
            self.ax_thrust.grid(True, alpha=0.3)
            self.ax_mass.grid(True, alpha=0.3)
            self.figure.tight_layout(pad=2.0)
            self.canvas.draw_idle()

            peak_thrust = float(np.max(thrust))
            burn_time = float(times[-1])
            initial_mass = float(masses[0])
            final_mass = float(masses[-1])
            self.summary_var.set(
                f"Peak thrust: {peak_thrust:.1f} N\n"
                f"Simulated time: {burn_time:.3f} s\n"
                f"Fuel consumed: {max(initial_mass - final_mass, 0.0):.4f} kg"
            )
            self.status_var.set("Simulation complete.")
        except Exception as exc:
            self.latest_velocity_profile = None
            self.status_var.set(f"Simulation failed: {exc}")
            print(traceback.format_exc())
            print(sys.exc_info()[2])


def main() -> None:
    root = tk.Tk()
    MotorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
