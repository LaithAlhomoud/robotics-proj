import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import mplcursors


# ============= Utility ===================================

def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ============= Robot model (differential drive) ==========

@dataclass
class DiffDriveRobot:
    """
    Unicycle model:
        x_dot = v cos(theta)
        y_dot = v sin(theta)
        theta_dot = omega
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # rad

    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])

    def step(self, v: float, omega: float, dt: float):
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = wrap_angle(self.theta + omega * dt)


# ============= PID controller ============================

@dataclass
class PID:
    """
    u(t) = Kp e(t) + Ki ∫e dt + Kd de/dt
    """
    Kp: float
    Ki: float
    Kd: float
    integral: float = 0.0
    prev_error: float = 0.0
    first: bool = True

    last_up: float = 0.0
    last_ui: float = 0.0
    last_ud: float = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.first = True
        self.last_up = self.last_ui = self.last_ud = 0.0

    def __call__(self, error: float, dt: float) -> float:
        self.integral += error * dt

        if self.first:
            derivative = 0.0
            self.first = False
        else:
            derivative = (error - self.prev_error) / max(dt, 1e-6)

        self.prev_error = error

        up = self.Kp * error
        ui = self.Ki * self.integral
        ud = self.Kd * derivative

        self.last_up, self.last_ui, self.last_ud = up, ui, ud
        return up + ui + ud


# ============= Reference trajectories ====================

def reference_line(t: float, v_ref: float = 0.5) -> Tuple[float, float, float, float, float]:
    """Straight line along x-axis."""
    x_d = v_ref * t
    y_d = 0.0
    theta_d = 0.0
    v_d = v_ref
    omega_d = 0.0
    return x_d, y_d, theta_d, v_d, omega_d


def reference_circle(t: float, R: float = 1.0, w_circ: float = 0.3) -> Tuple[float, float, float, float, float]:
    """Circular trajectory of radius R centered at origin."""
    x_d = R * np.cos(w_circ * t)
    y_d = R * np.sin(w_circ * t)

    dx_dt = -R * w_circ * np.sin(w_circ * t)
    dy_dt = R * w_circ * np.cos(w_circ * t)
    theta_d = np.arctan2(dy_dt, dx_dt)

    v_d = np.hypot(dx_dt, dy_dt)  # = R * w_circ
    omega_d = w_circ
    return x_d, y_d, theta_d, v_d, omega_d


# ============= Data container ============================

@dataclass
class SimData:
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)

    x_d: List[float] = field(default_factory=list)
    y_d: List[float] = field(default_factory=list)
    theta_d: List[float] = field(default_factory=list)

    ex: List[float] = field(default_factory=list)
    ey: List[float] = field(default_factory=list)
    e_theta: List[float] = field(default_factory=list)

    e_robot: List[float] = field(default_factory=list)

    v: List[float] = field(default_factory=list)
    omega: List[float] = field(default_factory=list)
    v_ref: List[float] = field(default_factory=list)
    omega_ref: List[float] = field(default_factory=list)

    v_up: List[float] = field(default_factory=list)
    v_ui: List[float] = field(default_factory=list)
    v_ud: List[float] = field(default_factory=list)

    w_up: List[float] = field(default_factory=list)
    w_ui: List[float] = field(default_factory=list)
    w_ud: List[float] = field(default_factory=list)


# ============= Simulation core  ================

def simulate(path: str = "line",
             controller_mode: str = "PID",
             closed_loop: bool = True,
             T: float = 20.0,
             dt: float = 0.01,
             pid_v_params=None,
             pid_w_params=None) -> SimData:
    """
    Simulate a differential-drive robot.
    path ∈ {"line", "circle"}
    controller_mode ∈ {"open_loop", "P", "PI", "PID"}
    """

    # initial pose with offset (to show convergence)
    robot = DiffDriveRobot(
        x=0.2,
        y=-0.3,
        theta=np.deg2rad(20)
    )

    # === Controller gains ===
    mode = controller_mode.upper()

    # If gains are explicitly provided, always use them
    if pid_v_params is not None and pid_w_params is not None:
        Kp_v, Ki_v, Kd_v = pid_v_params
        Kp_w, Ki_w, Kd_w = pid_w_params
        pid_v = PID(Kp=Kp_v, Ki=Ki_v, Kd=Kd_v)
        pid_w = PID(Kp=Kp_w, Ki=Ki_w, Kd=Kd_w)
    else:
        # Fallback (in case you ever call simulate without params)
        if mode == "P":
            print('fallback values used')
            pid_v = PID(Kp=0.8, Ki=0.0, Kd=0.0)
            pid_w = PID(Kp=0.7, Ki=0.0, Kd=0.0)
        elif mode == "PI":
            print('fallback values used')
            pid_v = PID(Kp=1.0, Ki=0.01, Kd=0.0)
            pid_w = PID(Kp=0.6, Ki=0.02, Kd=0.0)
        else:  # PID default
            print('fallback values used')
            pid_v = PID(Kp=0.9, Ki=0.03, Kd=0.1)
            pid_w = PID(Kp=0.5, Ki=0.04, Kd=0.1)

    pid_v.reset()
    pid_w.reset()

    data = SimData()
    steps = int(T / dt)

    for k in range(steps + 1):
        t = k * dt

        # desired trajectory
        if path == "line":
            x_d, y_d, theta_d, v_d, omega_d = reference_line(t)
        else:
            x_d, y_d, theta_d, v_d, omega_d = reference_circle(t)

        # error vector in world frame
        ex = x_d - robot.x
        ey = y_d - robot.y
        e_theta = wrap_angle(theta_d - robot.theta)

        # ----- Robot-frame error and combined e_robot -----
        cR, sR = np.cos(robot.theta), np.sin(robot.theta)
        ex_R = cR * ex + sR * ey
        ey_R = -sR * ex + cR * ey

        pos_err_R = np.hypot(ex_R, ey_R)

        # Combined error: position + weighted heading
        w_theta = 0.5  # weight on heading error
        e_robot = pos_err_R + w_theta * abs(e_theta)

        # distance error along robot heading (for v control)
        c, s = np.cos(robot.theta), np.sin(robot.theta)
        e_parallel = c * ex + s * ey
        dist_error = np.hypot(ex, ey) * np.sign(e_parallel)

        if closed_loop and controller_mode.lower() != "open_loop":
            # feedforward + PID correction
            v_cmd = v_d + pid_v(dist_error, dt)
            w_cmd = omega_d + pid_w(e_theta, dt)
        else:
            # pure open-loop
            v_cmd = v_d
            w_cmd = omega_d

        # saturate
        v_cmd = float(np.clip(v_cmd, -1.0, 1.0))
        w_cmd = float(np.clip(w_cmd, -2.0, 2.0))

        # log
        data.t.append(t)
        data.x.append(robot.x)
        data.y.append(robot.y)
        data.theta.append(robot.theta)

        data.x_d.append(x_d)
        data.y_d.append(y_d)
        data.theta_d.append(theta_d)

        data.ex.append(ex)
        data.ey.append(ey)
        data.e_theta.append(e_theta)
        data.e_robot.append(e_robot)

        data.v.append(v_cmd)
        data.omega.append(w_cmd)
        data.v_ref.append(v_d)
        data.omega_ref.append(omega_d)

        data.v_up.append(pid_v.last_up)
        data.v_ui.append(pid_v.last_ui)
        data.v_ud.append(pid_v.last_ud)

        data.w_up.append(pid_w.last_up)
        data.w_ui.append(pid_w.last_ui)
        data.w_ud.append(pid_w.last_ud)

        # update robot
        robot.step(v_cmd, w_cmd, dt)

    return data


# ============= AI tuner (PID only) =======================

def auto_tune_pid(path: str = "circle",
                  trials: int = 1000,
                  T: float = 20.0,
                  dt: float = 0.01):
    """
    Simple AI-like tuner for PID gains.
    Tries many random (Kp, Ki, Kd) values for v and w,
    runs the simulation, and keeps the set with the
    smallest mean squared combined error e_robot.
    """

    rng = np.random.default_rng(seed=0)

    best_cost = np.inf
    best_v = None
    best_w = None

    for i in range(trials):
        # --- sample random gains in reasonable ranges ---
        Kp_v = rng.uniform(0.1, 2.0)
        Ki_v = rng.uniform(0.0, 0.2)
        Kd_v = rng.uniform(0.0, 0.5)

        Kp_w = rng.uniform(0.5, 5.0)
        Ki_w = rng.uniform(0.0, 0.2)
        Kd_w = rng.uniform(0.0, 0.7)

        pid_v_params = (Kp_v, Ki_v, Kd_v)
        pid_w_params = (Kp_w, Ki_w, Kd_w)

        # run one simulation with these gains
        data = simulate(path=path,
                        controller_mode="PID",
                        closed_loop=True,
                        T=T,
                        dt=dt,
                        pid_v_params=pid_v_params,
                        pid_w_params=pid_w_params)

        e_robot = np.array(data.e_robot)
        cost = np.mean(e_robot ** 2)

        if cost < best_cost:
            best_cost = cost
            best_v = pid_v_params
            best_w = pid_w_params
            print(f"New best ({path}) #{i}: cost={best_cost:.5f}, "
                  f"v={best_v}, w={best_w}")

    return best_v, best_w, best_cost


# helper: derive P/PI/PID params from tuned PID
def derive_params_from_pid(best_v, best_w, mode: str):
    """
    mode ∈ {'P','PI','PID'}
    best_v = (Kp_v, Ki_v, Kd_v)
    best_w = (Kp_w, Ki_w, Kd_w)
    """
    Kp_v, Ki_v, Kd_v = best_v
    Kp_w, Ki_w, Kd_w = best_w

    if mode.upper() == "P":
        return (Kp_v, 0.0, 0.0), (Kp_w, 0.0, 0.0)
    elif mode.upper() == "PI":
        return (Kp_v, Ki_v, 0.0), (Kp_w, Ki_w, 0.0)
    else:  # PID
        return best_v, best_w


# ============= 2D plots (interactive) ====================

def plot_results(data: SimData,
                 path: str,
                 controller_mode: str,
                 closed_loop: bool = True):
    label_mode = controller_mode.upper()
    if not closed_loop or controller_mode.lower() == "open_loop":
        title = f"{path.capitalize()} path – OPEN-LOOP"
    else:
        title = f"{path.capitalize()} path – {label_mode} closed-loop"

    t = np.array(data.t)
    x = np.array(data.x)
    y = np.array(data.y)
    x_d = np.array(data.x_d)
    y_d = np.array(data.y_d)
    e_robot = np.array(data.e_robot)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # trajectory
    ax0 = axes[0]
    ax0.plot(x_d, y_d, '--', linewidth=2, label="Desired trajectory")
    ax0.plot(x, y, linewidth=2, label="Actual trajectory")
    ax0.set_xlabel("x [m]")
    ax0.set_ylabel("y [m]")
    ax0.set_title("Trajectory in x–y plane")
    ax0.axis("equal")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # errors
    ax1 = axes[1]
    ax1.plot(t, e_robot, linewidth=2.0, label="e_robot (combined)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Error")
    ax1.set_title("Combined error e_robot vs time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ====== INTERACTIVE CURSOR PART ======
    cursor = mplcursors.cursor(fig, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        artist = sel.artist
        x_data, y_data = sel.target[0], sel.target[1]
        ax = artist.axes

        if ax is ax0:
            sel.annotation.set_text(
                f"{artist.get_label()}\n"
                f"x = {x_data:.3f} m\n"
                f"y = {y_data:.3f} m"
            )
        else:
            sel.annotation.set_text(
                f"{artist.get_label()}\n"
                f"t = {x_data:.3f} s\n"
                f"value = {y_data:.4f}"
            )
        sel.annotation.get_bbox_patch().set_alpha(0.85)

    plt.show()


def plot_erobot_comparison(results_dict, path: str):
    plt.figure(figsize=(8, 5))
    plt.title(f"Combined error e_robot over time – {path.capitalize()} path")
    plt.xlabel("Time [s]")
    plt.ylabel("e_robot")

    for mode, data in results_dict.items():
        t = np.array(data.t)
        e_robot = np.array(data.e_robot)
        plt.plot(t, e_robot, label=mode.upper())

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============= ✨ 3D error visualization ==================

def fancy_circle_3d(data: SimData,
                    controller_mode: str = "PID"):

    x = np.array(data.x)
    y = np.array(data.y)
    x_d = np.array(data.x_d)
    y_d = np.array(data.y_d)
    ex = np.array(data.ex)
    ey = np.array(data.ey)

    pos_err = np.hypot(ex, ey)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_d, y_d, np.zeros_like(x_d),
            linestyle="--", linewidth=2, label="Desired trajectory")

    sc = ax.scatter(x, y, pos_err,
                    c=pos_err, cmap="plasma", s=15, label="Actual (error-coded)")

    cbar = fig.colorbar(sc, pad=0.1)
    cbar.set_label("Position error ‖(e_x, e_y)‖ [m]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("Position error [m]")
    ax.set_title(f"Circle path – 3D error landscape ({controller_mode.upper()} control)")
    ax.legend()

    ax.view_init(elev=30, azim=-60)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============= Main =======================================

if __name__ == "__main__":
    # === 0) Auto-tune PID separately for line and circle ===
    print("=== Auto-tuning PID for LINE path ===")
    best_v_line, best_w_line, best_cost_line = auto_tune_pid(
        path="line", trials=8000, T=20.0, dt=0.01
    )
    print("\nLINE best PID gains:")
    print("  v-loop (Kp, Ki, Kd):", best_v_line)
    print("  w-loop (Kp, Ki, Kd):", best_w_line)
    print("  cost:", best_cost_line)

    print("\n=== Auto-tuning PID for CIRCLE path ===")
    best_v_circle, best_w_circle, best_cost_circle = auto_tune_pid(
        path="circle", trials=5000, T=25.0, dt=0.01
    )
    print("\nCIRCLE best PID gains:")
    print("  v-loop (Kp, Ki, Kd):", best_v_circle)
    print("  w-loop (Kp, Ki, Kd):", best_w_circle)
    print("  cost:", best_cost_circle)

    # === 1) LINE: P, PI, PID derived from tuned PID ===
    line_results = {}
    for mode in ["open_loop", "P", "PI", "PID"]:
        v_params, w_params = derive_params_from_pid(best_v_line, best_w_line, mode)
        data = simulate(path="line",
                        controller_mode=mode,
                        closed_loop=True,
                        T=20.0,
                        dt=0.01,
                        pid_v_params=v_params if mode != "open_loop" else None,
                        pid_w_params=w_params if mode != "open_loop" else None)
        plot_results(data, path="line", controller_mode=mode, closed_loop=True)
        line_results[mode] = data

    # === 2) CIRCLE: P, PI, PID derived from tuned PID ===
    circle_results = {}
    for mode in ["open_loop", "P", "PI", "PID"]:
        v_params, w_params = derive_params_from_pid(best_v_circle, best_w_circle, mode)
        data = simulate(path="circle",
                        controller_mode=mode,
                        closed_loop=True,
                        T=25.0,
                        dt=0.01,
                        pid_v_params=v_params if mode != "open_loop" else None,
                        pid_w_params=w_params if mode != "open_loop" else None)
        plot_results(data, path="circle", controller_mode=mode, closed_loop=True)
        circle_results[mode] = data

    # Optional: 3D view for the best controller on circle (PID)
    fancy_circle_3d(circle_results["PID"], controller_mode="PID")

    # === 3) Compare combined error e_robot across P / PI / PID ===
    plot_erobot_comparison(line_results, path="line")
    plot_erobot_comparison(circle_results, path="circle")
