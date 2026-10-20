"""Bridges the simulator's vehicle config to the controller's own parameter set."""

from dataclasses import dataclass

import numpy as np

from hopper.config import VehicleConfig
from hopper.control.mission import MissionController, MissionParams
from hopper.control.params import ControlParams
from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment, Wind
from hopper.dynamics.mass_model import mass_properties
from hopper.estimated_controller import EstimatedController
from hopper.estimation.ekf import EkfParams
from hopper.sim.events import Outcome
from hopper.types import make_state


def control_params_from_config(cfg: VehicleConfig) -> ControlParams:
    return ControlParams(
        thrust_max=cfg.engine.thrust_max,
        thrust_min_frac=cfg.engine.thrust_min_frac,
        gimbal_max=cfg.engine.gimbal_max,
        gimbal_rate_max=cfg.engine.gimbal_rate_max,
        max_roll_torque=cfg.rcs.max_roll_torque,
        inertia_diag_dry=tuple(cfg.mass.dry_inertia),
        dry_mass=cfg.mass.dry_mass,
        dry_com_z=cfg.mass.dry_com_z,
        prop_mass_initial=cfg.tank.prop_mass_initial,
        tank_bottom_z=cfg.tank.bottom_z,
        tank_height=cfg.tank.height,
        tank_radius=cfg.tank.radius,
        leg_length=cfg.legs.length,
    )


_MIN_SIGMA = 1e-4  # measurement noise floor: a noise-free sensor must not make S singular


def ekf_params_from_config(cfg: VehicleConfig) -> EkfParams:
    """The filter's noise model, taken from the sensor config (what a designer would know)."""
    s = cfg.sensors
    imu = s.imu
    floor = lambda x: max(x, _MIN_SIGMA)  # noqa: E731
    return EkfParams(
        gyro_sigma=imu.gyro_noise_density * imu.rate**0.5,
        accel_sigma=imu.accel_noise_density * imu.rate**0.5,
        gyro_bias_rw=imu.gyro_bias_rw,
        accel_bias_rw=imu.accel_bias_rw,
        gyro_bias_init=imu.gyro_bias_init,
        accel_bias_init=imu.accel_bias_init,
        imu_delay=imu.latency + 0.5 / imu.rate,
        baro_sigma=floor(s.baro.noise_sigma),
        baro_bias_rw=s.baro.bias_rw,
        baro_latency=s.baro.latency,
        gps_pos_sigma=(
            floor(s.gps.pos_sigma_h),
            floor(s.gps.pos_sigma_h),
            floor(s.gps.pos_sigma_v),
        ),
        gps_vel_sigma=floor(s.gps.vel_sigma),
        gps_latency=s.gps.latency,
        lidar_sigma_abs=floor(s.lidar.sigma_abs),
        lidar_sigma_rel=s.lidar.sigma_rel,
        lidar_latency=s.lidar.latency,
        dry_mass=cfg.mass.dry_mass,
        prop_mass_initial=cfg.tank.prop_mass_initial,
        thrust_max=cfg.engine.thrust_max,
        thrust_min_frac=cfg.engine.thrust_min_frac,
        isp=cfg.engine.isp,
        throttle_tau=cfg.engine.throttle_tau,
    )


@dataclass(frozen=True)
class HopSummary:
    outcome: Outcome
    apogee: float  # m, max foot-tip height
    landing_error: float  # m, horizontal distance from the commanded site
    touchdown_speed: float  # m/s, vertical speed just before first contact
    touchdown_tilt: float  # rad, at the end of the run
    flight_time: float  # s, from first lift-off to first touchdown
    fuel_used: float  # kg
    t_end: float
    touchdown_hspeed: float = 0.0  # m/s, horizontal speed just before first contact
    max_tilt: float = 0.0  # rad, over the whole run
    max_gimbal: float = 0.0  # rad, largest |gimbal angle| on either axis
    touchdown_index: int = 0  # log row of the first touchdown


def run_hop(
    cfg: VehicleConfig,
    mission: MissionParams | None = None,
    seed: int = 0,
    command_delay: float = 0.0,
    wind_mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    t_end: float | None = None,
    log_decimation: int = 10,
    perfect_state: bool = True,
    ekf_params: EkfParams | None = None,
):
    """Fly the hop. Returns `(SimResult, HopSummary)`.

    `perfect_state=False` flies on the EKF estimate built from the sensors (`ekf_params` default to
    `ekf_params_from_config(cfg)`) and adds `est_*` / `sig_*` columns to the log.
    """
    mission = mission or MissionParams()
    rng = np.random.default_rng(seed)
    com_z = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z
    # Start with the legs preloaded by ~the vehicle weight, so the pad counts as ground contact.
    sag = 0.01
    x0 = make_state(
        [0.0, 0.0, com_z + cfg.legs.length - sag],
        [0, 0, 0],
        None,
        [0, 0, 0],
        cfg.tank.prop_mass_initial,
    )
    env = Environment(wind=Wind(np.array(wind_mean)))
    controller = MissionController(control_params_from_config(cfg), mission)
    if not perfect_state:
        controller = EstimatedController(ekf_params or ekf_params_from_config(cfg), controller)
    sim = DrakeSimulation(
        cfg,
        controller,
        env,
        x0,
        rng,
        perfect_state=perfect_state,
        command_delay=command_delay,
        log_decimation=log_decimation,
    )
    res = sim.run(t_end)
    return res, summarize_hop(res, cfg, mission)


def summarize_hop(
    res, cfg: VehicleConfig, mission: MissionParams, origin_xy: tuple[float, float] = (0.0, 0.0)
) -> HopSummary:
    log = res.log
    prop = log["prop"].to_numpy()
    com = np.array([mass_properties(p, cfg).com_z for p in prop])
    foot = log["pos_z"].to_numpy() - com - cfg.legs.length
    q_all = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy()
    q = q_all[-1]
    tilt = float(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1.0, 1.0)))
    last = log.iloc[-1]
    err = np.hypot(
        last.pos_x - origin_xy[0] - mission.target_xy[0],
        last.pos_y - origin_xy[1] - mission.target_xy[1],
    )

    airborne = np.flatnonzero(foot > 0.1)
    contact = np.flatnonzero((log["n_contacts"].to_numpy() > 0) & (np.arange(len(log)) > 0))
    first_air = airborne[0] if len(airborne) else 0
    later = contact[contact > first_air] if len(contact) else contact
    k_td = int(later[0]) if len(later) else len(log) - 1
    return HopSummary(
        outcome=res.outcome,
        apogee=float(foot.max()),
        landing_error=float(err),
        touchdown_speed=float(-log["vel_z"].iloc[max(k_td - 1, 0)]),
        touchdown_tilt=tilt,
        flight_time=float(log["t"].iloc[k_td] - log["t"].iloc[first_air]),
        fuel_used=float(prop[0] - prop[-1]),
        t_end=res.t_end,
        touchdown_hspeed=float(
            np.hypot(log["vel_x"].iloc[max(k_td - 1, 0)], log["vel_y"].iloc[max(k_td - 1, 0)])
        ),
        max_tilt=float(np.max(2 * np.arcsin(np.clip(np.hypot(*q_all[:, 1:3].T), 0.0, 1.0)))),
        max_gimbal=float(np.abs(log[["gim_x", "gim_y"]].to_numpy()).max()),
        touchdown_index=k_td,
    )
