"""Quasi-steady drag as a Drake LeafSystem .

The force acts at the centre of pressure on the body z axis, measured from the pivot (the body
origin), so Drake produces the weathercock torque about the CoM itself. Air-relative velocity and
ISA density use the CoM; its offset comes from the propellant mass.
"""

import math

import numpy as np
from pydrake.all import AbstractValue, ExternallyAppliedSpatialForce, LeafSystem, SpatialForce
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.state import BODY_NAME
from hopper.dynamics.environment import atmosphere
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import cross, quat_to_dcm


class AeroLoads(LeafSystem):
    def __init__(self, cfg: VehicleConfig, plant: MultibodyPlant, aero_enabled: bool = True):
        super().__init__()
        self._cfg = cfg
        self._aero = cfg.aero
        self._enabled = aero_enabled
        self._body_index = plant.GetBodyByName(BODY_NAME).index()
        self._state_port = self.DeclareVectorInputPort("plant_state", plant.num_multibody_states())
        self._wind_port = self.DeclareVectorInputPort("wind", 3)
        self._prop_port = self.DeclareVectorInputPort("propellant", 1)
        self._force_port = self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc_forces,
        )

    def get_plant_state_input_port(self):
        return self._state_port

    def get_wind_input_port(self):
        return self._wind_port

    def get_propellant_input_port(self):
        return self._prop_port

    def get_force_output_port(self):
        return self._force_port

    def _calc_forces(self, context, out):
        a = self._aero
        force = ExternallyAppliedSpatialForce()
        force.body_index = self._body_index
        force.p_BoBq_B = np.array([0.0, 0.0, a.cp_z])
        f_w = np.zeros(3)
        if self._enabled:
            s = self._state_port.Eval(context)
            rot = quat_to_dcm(s[:4])
            com_z = mass_properties(self._prop_port.Eval(context)[0], self._cfg).com_z
            r_w = rot @ np.array([0.0, 0.0, com_z])
            p_com_z = s[6] + r_w[2]
            v_com = s[10:13] + cross(s[7:10], r_w)
            v_b = rot.T @ (v_com - self._wind_port.Eval(context))
            k = -0.5 * atmosphere(p_com_z).density * math.sqrt(v_b @ v_b)
            f_b = k * np.array(
                [
                    a.cd_lateral * a.area_lateral * v_b[0],
                    a.cd_lateral * a.area_lateral * v_b[1],
                    a.cd_axial * a.area_axial * v_b[2],
                ]
            )
            f_w = rot @ f_b
        force.F_Bq_W = SpatialForce(tau=np.zeros(3), f=f_w)
        out.set_value([force])
