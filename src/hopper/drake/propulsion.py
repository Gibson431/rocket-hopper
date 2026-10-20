"""Engine as a Drake LeafSystem: throttle lag, gimbal actuator, thrust and RCS force, mass flow.

Scalar math (no per-call Drake objects) because it runs in every RK stage (docs/DRAKE_DESIGN.md Q8).
Continuous state is the state-vector slots 13..18:
[prop, thrust fraction, gx, gy, gx_rate, gy_rate]. Thrust acts at the body origin, which is the
gimbal pivot, so Drake produces the torque about the CoM itself (Q2).
"""

import math

import numpy as np
from pydrake.all import (
    AbstractValue,
    BasicVector,
    ExternallyAppliedSpatialForce,
    LeafSystem,
    SpatialForce,
)
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.state import BODY_NAME
from hopper.dynamics.environment import G0
from hopper.math3d import quat_to_dcm

# continuous state indices
_PROP, _THR, _GX, _GY, _GXR, _GYR = range(6)
# Engine parameters live in the context (not Python attributes) so that changing them, as a fault
# does, invalidates cached outputs.
_PARAMS = (
    "thrust_max",
    "thrust_min_frac",
    "isp",
    "throttle_tau",
    "gimbal_max",
    "gimbal_rate_max",
    "gimbal_wn",
    "gimbal_zeta",
)


def _param_vector(cfg: VehicleConfig) -> np.ndarray:
    return np.array([getattr(cfg.engine, n) for n in _PARAMS] + [cfg.rcs.max_roll_torque])


class _Params:
    """Read-only view of the parameter vector with the config's field names."""

    __slots__ = (*_PARAMS, "max_roll_torque")

    def __init__(self, values):
        for name, v in zip((*_PARAMS, "max_roll_torque"), values, strict=True):
            setattr(self, name, float(v))


class Engine(LeafSystem):
    """Input command vector [throttle, gx, gy, roll_torque] (the Command fields, in that order).
    The command should come from discrete state (the controller host), never from a time-dependent
    source evaluated inside the RK stages (Q5 hazard)."""

    def __init__(self, cfg: VehicleConfig, plant: MultibodyPlant):
        super().__init__()
        self._body_index = plant.GetBodyByName(BODY_NAME).index()

        self.DeclareNumericParameter(BasicVector(_param_vector(cfg)))
        self._cmd_port = self.DeclareVectorInputPort("command", 4)
        self._state_port = self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        initial = np.zeros(6)
        initial[_PROP] = cfg.tank.prop_mass_initial
        self.DeclareContinuousState(BasicVector(initial), 0, 0, 6)
        self._force_port = self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )
        self._prop_port = self.DeclareVectorOutputPort("propellant", 1, self._calc_prop)
        self._thrust_port = self.DeclareVectorOutputPort("thrust", 1, self._calc_thrust)

    def get_command_input_port(self):
        return self._cmd_port

    def get_plant_state_input_port(self):
        return self._state_port

    def get_force_output_port(self):
        return self._force_port

    def get_propellant_output_port(self):
        return self._prop_port

    def get_thrust_output_port(self):
        return self._thrust_port

    def set_actuator_state(
        self, context, prop, thrust_frac=0.0, gimbal=(0.0, 0.0), rate=(0.0, 0.0)
    ):
        context.get_mutable_continuous_state_vector().SetFromVector(
            np.array([prop, thrust_frac, *gimbal, *rate], dtype=float)
        )

    def set_config(self, context, cfg: VehicleConfig) -> None:
        """Write the engine and RCS parameters of `cfg` (a parameter fault's effective config)."""
        context.get_mutable_numeric_parameter(0).SetFromVector(_param_vector(cfg))

    @staticmethod
    def _params(context) -> _Params:
        return _Params(context.get_numeric_parameter(0).CopyToVector())

    @staticmethod
    def _thrust(s, eng) -> float:
        return eng.thrust_max * s[_THR] if s[_PROP] > 0.0 else 0.0

    def _calc_prop(self, context, out):
        out.SetAtIndex(0, max(context.get_continuous_state_vector().GetAtIndex(_PROP), 0.0))

    def _calc_thrust(self, context, out):
        out.SetAtIndex(
            0,
            self._thrust(
                context.get_continuous_state_vector().CopyToVector(), self._params(context)
            ),
        )

    def _calc_forces(self, context, out):
        eng = self._params(context)
        s = context.get_continuous_state_vector().CopyToVector()
        roll = min(max(self._cmd_port.Eval(context)[3], -1.0), 1.0) * eng.max_roll_torque
        thrust = self._thrust(s, eng)
        # The physical angle never exceeds the limit, even if the integrator overshoots it.
        gx = min(max(s[_GX], -eng.gimbal_max), eng.gimbal_max)
        gy = min(max(s[_GY], -eng.gimbal_max), eng.gimbal_max)
        f_b = thrust * np.array(
            [math.cos(gx) * math.sin(gy), -math.sin(gx), math.cos(gx) * math.cos(gy)]
        )
        rot = quat_to_dcm(self._state_port.Eval(context)[:4])
        force = ExternallyAppliedSpatialForce()
        force.body_index = self._body_index
        force.p_BoBq_B = np.zeros(3)
        force.F_Bq_W = SpatialForce(tau=rot @ np.array([0.0, 0.0, roll]), f=rot @ f_b)
        out.set_value([force])

    def DoCalcTimeDerivatives(self, context, derivatives):  # noqa: N802 (Drake override)
        eng = self._params(context)
        s = context.get_continuous_state_vector().CopyToVector()
        cmd = self._cmd_port.Eval(context)
        has_prop = s[_PROP] > 0.0

        # 0 = off; any other request is clamped into the throttleable range.
        if cmd[0] <= 0.0 or not has_prop:
            thr_cmd = 0.0
        else:
            thr_cmd = min(max(cmd[0], eng.thrust_min_frac), 1.0)
        d = np.zeros(6)
        d[_PROP] = -self._thrust(s, eng) / (eng.isp * G0)
        d[_THR] = (thr_cmd - s[_THR]) / eng.throttle_tau

        for i, (g, r, c) in enumerate(((s[_GX], s[_GXR], cmd[1]), (s[_GY], s[_GYR], cmd[2]))):
            g_cmd = min(max(c, -eng.gimbal_max), eng.gimbal_max)
            rate = min(max(r, -eng.gimbal_rate_max), eng.gimbal_rate_max)
            acc = eng.gimbal_wn**2 * (g_cmd - g) - 2 * eng.gimbal_zeta * eng.gimbal_wn * r
            # Stop accelerating outward at the rate limit and at the angle limit.
            if abs(r) >= eng.gimbal_rate_max and acc * r > 0:
                acc = 0.0
            if abs(g) >= eng.gimbal_max and g * rate > 0:
                rate = 0.0
                if acc * g > 0:
                    acc = 0.0
            d[_GX + i] = rate
            d[_GXR + i] = acc
        derivatives.get_mutable_vector().SetFromVector(d)
