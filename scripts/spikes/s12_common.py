"""Shared spike helpers: legacy reference runs, a Drake hopper plant with a frame at the pivot."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pydrake.all import (  # noqa: E402
    AbstractValue, DiagramBuilder, ExternallyAppliedSpatialForce, LeafSystem, MultibodyPlant,
    Quaternion, RigidTransform, RotationMatrix, RotationalInertia, Simulator, SpatialForce,
    SpatialInertia, SpatialVelocity, ResetIntegratorFromFlags,
)

from hopper.config import load_vehicle  # noqa: E402
from hopper.dynamics.environment import Environment  # noqa: E402
from hopper.dynamics.mass_model import mass_properties  # noqa: E402
from hopper.dynamics.propulsion import propulsion  # noqa: E402
from hopper.dynamics.vehicle import vehicle_derivative  # noqa: E402
from hopper.math3d import quat_rotate, quat_to_dcm  # noqa: E402
from hopper.sim.integrator import rk4_step  # noqa: E402
from hopper.types import GIM, GIMR, N_STATE, OMEGA, POS, PROP, QUAT, THR, VEL, Command, make_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG = load_vehicle(ROOT / "configs" / "vehicle_default.yaml")
G = 9.80665


def schedule(t: float) -> Command:
    """Open-loop command, piecewise constant on whole seconds (aligned with every step grid)."""
    k = int(np.floor(t + 1e-9))
    thr = [0.6, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4, 0.4][min(k, 11)]
    gx = {1: 0.02, 2: -0.02, 5: -0.015, 6: 0.015}.get(k, 0.0)
    gy = {1: 0.01, 2: -0.01, 7: 0.02, 8: -0.02}.get(k, 0.0)
    roll = 0.3 if 3 <= k < 5 else 0.0
    return Command(throttle=thr, gimbal=np.array([gx, gy]), roll_torque=roll)


def initial_state(z0: float = 30.0) -> np.ndarray:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    x = make_state(pos=[0.0, 0.0, z0], vel=[0.5, -0.3, 1.0], quat=q, omega=[0.05, -0.04, 0.1],
                   prop=CFG.tank.prop_mass_initial)
    return x


def legacy_run(t_end, dt, x0=None, sched=schedule, env=None, sample_dt=0.5):
    """Legacy RK4 flight, command constant over each step. Returns {t: state} at sample times."""
    env = env or Environment(aero_enabled=False)
    x = (initial_state() if x0 is None else x0).copy()
    n = int(round(t_end / dt))
    every = int(round(sample_dt / dt))
    out = {0.0: x.copy()}
    for k in range(n):
        t = k * dt
        c = sched(t)
        x = rk4_step(lambda tt, xx: vehicle_derivative(tt, xx, c, CFG, env), t, x, dt)
        if (k + 1) % every == 0:
            out[round((k + 1) * dt, 9)] = x.copy()
    return out


# ---------------------------------------------------------------- Drake side

def spatial_inertia(prop):
    mp = mass_properties(prop, CFG)
    # central inertia (about the CoM, body axes) + CoM position from the pivot
    return SpatialInertia.MakeFromCentralInertia(
        mp.mass, np.array([0.0, 0.0, mp.com_z]),
        RotationalInertia(mp.inertia[0, 0], mp.inertia[1, 1], mp.inertia[2, 2]))


def com_z(prop):
    return mass_properties(prop, CFG).com_z


def legacy_to_drake(x):
    """19-state (CoM) -> (q[4], p_pivot, w_I, v_pivot). Pivot is the body origin."""
    q = x[QUAT]
    R = quat_to_dcm(q)
    r = R @ np.array([0.0, 0.0, com_z(x[PROP])])  # pivot -> CoM, inertial
    w_I = R @ x[OMEGA]
    p_o = x[POS] - r
    v_o = x[VEL] - np.cross(w_I, r)
    return q, p_o, w_I, v_o


def drake_to_legacy(q, p_o, w_I, v_o, prop):
    R = quat_to_dcm(q)
    r = R @ np.array([0.0, 0.0, com_z(prop)])
    return p_o + r, v_o + np.cross(w_I, r), R.T @ w_I


class Engine(LeafSystem):
    """Continuous-state engine (PROP, THR, GIM, GIMR) using the legacy `propulsion`, output as a
    spatial force at the body origin. Command comes from `sched(t)`."""

    def __init__(self, plant, body, sched=schedule, cmd_period=1.0):
        super().__init__()
        self._body_index = body.index()
        self._sched = sched
        self.DeclareContinuousState(6)
        self._state_in = self.DeclareVectorInputPort("plant_state", 13)
        # Command lives in discrete state: a time-dependent derivative would be stale in the
        # integrator's FSAL derivative cache across a switch (seen as a 1-stage lag).
        self.DeclareDiscreteState(4)
        self.DeclarePeriodicDiscreteUpdateEvent(cmd_period, 0.0, self._update_cmd)
        self.DeclareVectorOutputPort("prop", 1, lambda c, o: o.SetFromVector([c.get_continuous_state_vector().GetAtIndex(0)]))
        self.DeclareAbstractOutputPort(
            "force", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self._calc_force)

    def _update_cmd(self, c, ds):
        cmd = self._sched(c.get_time())
        ds.set_value(np.array([cmd.throttle, *cmd.gimbal, cmd.roll_torque]))

    def _cmd(self, c):
        v = c.get_discrete_state_vector().CopyToVector()
        return Command(throttle=v[0], gimbal=v[1:3], roll_torque=v[3])

    def _legacy_x(self, c):
        s = c.get_continuous_state_vector().CopyToVector()
        x = np.zeros(N_STATE)
        x[PROP], x[THR], x[GIM], x[GIMR] = s[0], s[1], s[2:4], s[4:6]
        return x

    def DoCalcTimeDerivatives(self, c, d):
        x = self._legacy_x(c)
        cmd = self._cmd(c)
        mp = mass_properties(x[PROP], CFG)
        p = propulsion(x, cmd, CFG, mp)
        xd = p.xdot_act
        d.get_mutable_vector().SetFromVector(np.concatenate([[xd[PROP], xd[THR]], xd[GIM], xd[GIMR]]))

    def _calc_force(self, c, out):
        x = self._legacy_x(c)
        cmd = self._cmd(c)
        mp = mass_properties(x[PROP], CFG)
        p = propulsion(x, cmd, CFG, mp)
        f = ExternallyAppliedSpatialForce()
        f.body_index = self._body_index
        f.p_BoBq_B = np.zeros(3)
        # propulsion's torque is about the CoM: remove r x F, keep the RCS roll torque.
        q = self._state_in.Eval(c)[:4]
        R = quat_to_dcm(q)
        f.F_Bq_W = SpatialForce(R @ np.array([0.0, 0.0, p.torque_B[2]]), R @ p.force_B)
        out.set_value([f])


class DrakeHopper:
    """Plant (floating body, frame at the pivot) + engine, mass parameters updated between steps."""

    def __init__(self, dt, integrator="rk3", sched=schedule, x0=None, plant_dt=0.0, cmd_period=1.0):
        b = DiagramBuilder()
        self.plant = b.AddSystem(MultibodyPlant(plant_dt))
        self.body = self.plant.AddRigidBody("hopper", spatial_inertia(CFG.tank.prop_mass_initial))
        self.plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -G])  # Drake default is 9.81
        self.plant.Finalize()
        self.engine = b.AddSystem(Engine(self.plant, self.body, sched, cmd_period))
        b.Connect(self.engine.GetOutputPort("force"), self.plant.get_applied_spatial_force_input_port())
        b.Connect(self.plant.get_state_output_port(), self.engine.GetInputPort("plant_state"))
        self.diagram = b.Build()
        self.sim = Simulator(self.diagram)
        name = {"rk3": "runge_kutta3", "rk2": "runge_kutta2"}.get(integrator, integrator)
        self.integ = ResetIntegratorFromFlags(self.sim, name, dt)
        if integrator == "rk3":
            self.integ.set_fixed_step_mode(True)
        self.dt = dt
        self.ctx = self.sim.get_mutable_context()
        self.pctx = self.plant.GetMyMutableContextFromRoot(self.ctx)
        self.ectx = self.engine.GetMyMutableContextFromRoot(self.ctx)
        self._prop_set = CFG.tank.prop_mass_initial
        self.mid = False
        self.set_legacy_state(initial_state() if x0 is None else x0)
        self.sim.Initialize()

    def set_legacy_state(self, x):
        q, p_o, w_I, v_o = legacy_to_drake(x)
        self.body.SetSpatialInertiaInBodyFrame(self.pctx, spatial_inertia(x[PROP]))
        self._prop_set = x[PROP]
        self.plant.SetFreeBodyPose(self.pctx, self.body, RigidTransform(Quaternion(q), p_o))
        self.plant.SetFreeBodySpatialVelocity(self.pctx, self.body, SpatialVelocity(w_I, v_o))
        self.ectx.get_mutable_continuous_state_vector().SetFromVector(
            np.concatenate([[x[PROP], x[THR]], x[GIM], x[GIMR]]))

    def prop(self):
        return self.ectx.get_continuous_state_vector().GetAtIndex(0)

    def update_mass(self, mode="none"):
        """mode: none = keep the pivot state; vel = keep the CoM velocity; pos = keep CoM pos+vel
        (legacy semantics: the legacy state is the CoM and its motion is continuous)."""
        if mode != "none":
            st = self.plant.GetPositionsAndVelocities(self.pctx)
            prop_old = self._prop_set
            q, p_o, w_I, v_o = st[:4], st[4:7], st[7:10], st[10:13]
            p_com, v_com, _ = drake_to_legacy(q, p_o, w_I, v_o, prop_old)
        prop = self.prop()
        if self.mid:  # mass for the coming step = mass at its midpoint (removes the half-step lag)
            prop += 0.5 * self.dt * self.engine.EvalTimeDerivatives(self.ectx).get_vector().GetAtIndex(0)
        self.body.SetSpatialInertiaInBodyFrame(self.pctx, spatial_inertia(prop))
        self._prop_set = prop
        if mode != "none":
            R = quat_to_dcm(q)
            r = R @ np.array([0.0, 0.0, com_z(prop)])
            v_new = v_com - np.cross(w_I, r)
            p_new = p_com - r if mode == "pos" else p_o
            self.plant.SetFreeBodyPose(self.pctx, self.body, RigidTransform(Quaternion(q), p_new))
            self.plant.SetFreeBodySpatialVelocity(self.pctx, self.body, SpatialVelocity(w_I, v_new))

    def legacy_state(self):
        s = self.plant.GetPositionsAndVelocities(self.pctx)
        a = self.ectx.get_continuous_state_vector().CopyToVector()
        prop = a[0]
        p, v, w = drake_to_legacy(s[:4], s[4:7], s[7:10], s[10:13], prop)
        x = np.zeros(N_STATE)
        x[POS], x[VEL], x[QUAT], x[OMEGA] = p, v, s[:4] / np.linalg.norm(s[:4]), w
        x[PROP], x[THR], x[GIM], x[GIMR] = a[0], a[1], a[2:4], a[4:6]
        return x

    def run(self, t_end, sample_dt=0.5, mass_mode="none"):
        out = {0.0: self.legacy_state()}
        n = int(round(t_end / self.dt))
        every = int(round(sample_dt / self.dt))
        for k in range(n):
            self.sim.AdvanceTo((k + 1) * self.dt)
            self.update_mass(mass_mode)
            if (k + 1) % every == 0:
                out[round((k + 1) * self.dt, 9)] = self.legacy_state()
        return out


def compare(a, b):
    """Max abs differences (pos m, vel m/s, att rad, omega rad/s, prop kg) over common times."""
    worst = np.zeros(5)
    for t in sorted(set(a) & set(b)):
        xa, xb = a[t], b[t]
        qa, qb = xa[QUAT], xb[QUAT]
        ang = 2 * min(np.linalg.norm(qa - qb), np.linalg.norm(qa + qb))
        d = [np.linalg.norm(xa[POS] - xb[POS]), np.linalg.norm(xa[VEL] - xb[VEL]), ang,
             np.linalg.norm(xa[OMEGA] - xb[OMEGA]), abs(xa[PROP] - xb[PROP])]
        worst = np.maximum(worst, d)
    return worst
