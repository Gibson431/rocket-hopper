"""Sensors on the Drake diagram: the `SensorSuite` driven from the plant state.

One periodic update at the physics step samples every sensor and stores the delivered `SensorData`
as abstract state. The suite owns the noise generators; they are drawn only in this update, never
in an output callback (docs/DRAKE_DESIGN.md Q7). It is an unrestricted update on purpose: Drake
runs unrestricted updates before the discrete ones at the same time, so the host's update at t_k
reads the samples taken at t_k (a test pins this).
"""

import copy
from collections.abc import Sequence

import numpy as np
from hopper_core.math3d import cross, quat_to_dcm
from hopper_core.types import N_STATE, OMEGA, POS, PROP, QUAT, VEL, SensorData
from pydrake.all import AbstractValue, LeafSystem
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.controller_host import estimate_from_plant_state
from hopper.drake.state import BODY_NAME
from hopper.dynamics.mass_model import mass_properties
from hopper.sensors.suite import SensorSuite
from hopper.sim.faults import Fault

NO_SAMPLES = SensorData(0.0, None, None, None, None, None, None)


class SensorsSystem(LeafSystem):
    """Inputs `plant_state` (13), `propellant` (1) and `accel` (the plant's body spatial
    accelerations); output `sensors` (a `SensorData`: the samples delivered at the last update).
    `faults` act on the delivered samples (`modify_sensors`)."""

    def __init__(
        self,
        cfg: VehicleConfig,
        plant: MultibodyPlant,
        rng: np.random.Generator,
        faults: Sequence[Fault] = (),
    ):
        super().__init__()
        self._cfg = cfg
        self._dt = cfg.sim.dt
        self._faults = tuple(faults)
        self._gravity = np.asarray(plant.gravity_field().gravity_vector(), dtype=float)
        self._body_index = plant.GetBodyByName(BODY_NAME).index()
        # The suite spawns child generators from `rng`; keep a pristine copy to rebuild per run.
        self._rng0 = copy.deepcopy(rng)
        self._suite = SensorSuite(cfg, rng)

        self._state_port = self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        self._prop_port = self.DeclareVectorInputPort("propellant", 1)
        self._accel_port = self.DeclareAbstractInputPort(
            "accel", plant.get_body_spatial_accelerations_output_port().Allocate()
        )
        state_index = self.DeclareAbstractState(AbstractValue.Make(NO_SAMPLES))
        self.DeclarePeriodicUnrestrictedUpdateEvent(self._dt, 0.0, self._update)
        self._out_port = self.DeclareAbstractOutputPort(
            "sensors",
            lambda: AbstractValue.Make(NO_SAMPLES),
            lambda context, out: out.set_value(context.get_abstract_state(0).get_value()),
            prerequisites_of_calc={self.abstract_state_ticket(state_index)},
        )

    def get_plant_state_input_port(self):
        return self._state_port

    def get_propellant_input_port(self):
        return self._prop_port

    def get_accel_input_port(self):
        return self._accel_port

    def get_sensors_output_port(self):
        return self._out_port

    def reset(self) -> None:
        """Fresh noise streams, biases and delay queues: the state of a new run."""
        self._suite = SensorSuite(self._cfg, copy.deepcopy(self._rng0))

    def specific_force(self, context) -> np.ndarray:
        """Accelerometer input at the CoM, body axes: `R^T (a_com - g)` (Q6)."""
        s = self._state_port.Eval(context)
        prop = self._prop_port.Eval(context)[0]
        est = estimate_from_plant_state(s, prop, self._cfg, 0.0)
        rot = quat_to_dcm(est.quat)
        a_wb = self._accel_port.Eval(context)[int(self._body_index)]
        w = s[7:10]
        r = rot @ np.array([0.0, 0.0, mass_properties(prop, self._cfg).com_z])
        a_com = a_wb.translational() + cross(a_wb.rotational(), r) + cross(w, cross(w, r))
        return rot.T @ (a_com - self._gravity)

    def _update(self, context, state):
        t = round(context.get_time() / self._dt) * self._dt
        s = self._state_port.Eval(context)
        prop = self._prop_port.Eval(context)[0]
        est = estimate_from_plant_state(s, prop, self._cfg, t)
        x = np.zeros(N_STATE)
        x[POS], x[VEL], x[QUAT], x[OMEGA], x[PROP] = est.pos, est.vel, est.quat, est.omega, prop
        info = {}
        if self._suite.needs_info(t):
            info["specific_force_B"] = self.specific_force(context)
        data = self._suite.sample(t, x, info)
        for fault in self._faults:
            data = fault.modify_sensors(t, data)
        state.get_mutable_abstract_state(0).set_value(data)
