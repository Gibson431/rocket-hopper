"""Ground and leg contact on the Drake plant, plus the observer that reports it.

Contact model (docs/DRAKE_DESIGN.md Q4): Drake point contact between a half-space ground and one
small sphere per foot. Parameter mapping from `LegsConfig`:

- `count`, `radius`, `length`: foot i sits at `[r cos a_i, r sin a_i, -length]` from the gimbal
  pivot (the body origin), `a_i = 2 pi i / count`, exactly the legacy foot positions. The sphere
  centre is one sphere radius above the foot point, so the contact point is the foot point.
- `stiffness`: ground and foot each get `2 k`, so the series value is `k` (Drake combines point
  contact stiffnesses in series).
- `damping`: not used. Drake's Hunt-Crossley dissipation (s/m) scales the stiffness force instead of
  being a fixed N s/m; `DISSIPATION` is tuned so a drop does not rebound (the legacy damper is not
  reproduced).
- `friction`: Coulomb coefficient on both geometries; `STICTION_TOLERANCE` is the slip speed of full
  friction (matches the legacy regularisation `V_EPS`).

The engine base has no collision geometry: a base strike ends the run as a crash, so the observer
only flags it (pivot below the ground, as the legacy `base_strike`).
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import (
    AbstractValue,
    AddContactMaterial,
    ContactResults,
    CoulombFriction,
    HalfSpace,
    LeafSystem,
    ProximityProperties,
    RigidTransform,
    Sphere,
)
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.state import BODY_NAME

DISSIPATION = 2.5  # s/m, Hunt-Crossley; lands from 0.5 m without a rebound
STICTION_TOLERANCE = 0.05  # m/s
FOOT_RADIUS = 0.01  # m, the collision sphere


@dataclass(frozen=True)
class LegGeometry:
    """Ids of the foot collision geometries, in foot order, for the observer."""

    foot_ids: tuple


def foot_points(cfg: VehicleConfig) -> np.ndarray:
    """(count, 3) foot points from the gimbal pivot, body frame."""
    legs = cfg.legs
    a = 2 * np.pi * np.arange(legs.count) / legs.count
    return np.stack(
        [legs.radius * np.cos(a), legs.radius * np.sin(a), np.full(legs.count, -legs.length)],
        axis=1,
    )


def add_legs(plant: MultibodyPlant, scene_graph, cfg: VehicleConfig) -> LegGeometry:
    """Add the ground and the foot spheres. Call before `plant.Finalize()`."""
    if scene_graph is None:
        raise ValueError("add_legs needs a plant built with a scene graph (pass a builder)")

    def props() -> ProximityProperties:
        p = ProximityProperties()
        mu = CoulombFriction(cfg.legs.friction, cfg.legs.friction)
        AddContactMaterial(DISSIPATION, 2.0 * cfg.legs.stiffness, mu, p)
        return p

    plant.set_stiction_tolerance(STICTION_TOLERANCE)
    plant.RegisterCollisionGeometry(
        plant.world_body(), RigidTransform(), HalfSpace(), "ground", props()
    )
    body = plant.GetBodyByName(BODY_NAME)
    ids = tuple(
        plant.RegisterCollisionGeometry(
            body,
            RigidTransform(p + np.array([0.0, 0.0, FOOT_RADIUS])),
            Sphere(FOOT_RADIUS),
            f"foot{i}",
            props(),
        )
        for i, p in enumerate(foot_points(cfg))
    )
    return LegGeometry(foot_ids=ids)


class ContactObserver(LeafSystem):
    """Reports the ground contact of the feet.

    Inputs: `plant_state` (13) and `contact_results` (the plant's). Outputs: `foot_forces` (n,
    normal force per foot, N, ordered as the legs), `contact_count` (1, feet touching),
    `total_force` (3, world frame, force of the ground on the vehicle) and `base_strike` (1, 1.0
    when the gimbal pivot is below the ground).
    """

    def __init__(self, plant: MultibodyPlant, legs: LegGeometry):
        super().__init__()
        self._body_index = plant.GetBodyByName(BODY_NAME).index()
        self._slot = {gid: i for i, gid in enumerate(legs.foot_ids)}
        self._n = len(legs.foot_ids)
        self._state = self.DeclareVectorInputPort("plant_state", 13)
        self._results = self.DeclareAbstractInputPort(
            "contact_results", AbstractValue.Make(ContactResults())
        )
        self._foot_forces = self.DeclareVectorOutputPort("foot_forces", self._n, self._calc_feet)
        self._count = self.DeclareVectorOutputPort("contact_count", 1, self._calc_count)
        self._total = self.DeclareVectorOutputPort("total_force", 3, self._calc_total)
        self._strike = self.DeclareVectorOutputPort("base_strike", 1, self._calc_strike)

    def get_plant_state_input_port(self):
        return self._state

    def get_contact_results_input_port(self):
        return self._results

    def get_foot_forces_output_port(self):
        return self._foot_forces

    def get_contact_count_output_port(self):
        return self._count

    def get_total_force_output_port(self):
        return self._total

    def get_base_strike_output_port(self):
        return self._strike

    def _contacts(self, context):
        """(per-foot normal force, total world force on the vehicle)."""
        results = self._results.Eval(context)
        feet, total = np.zeros(self._n), np.zeros(3)
        for k in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(k)
            # contact_force is the force on body B; flip it when the vehicle is body A.
            sign = 1.0 if info.bodyB_index() == self._body_index else -1.0
            f = sign * np.asarray(info.contact_force())
            pair = info.point_pair()
            for gid in (pair.id_A, pair.id_B):
                if gid in self._slot:
                    feet[self._slot[gid]] += max(f[2], 0.0)  # flat ground: z is the normal
            total += f
        return feet, total

    def _calc_feet(self, context, out):
        out.SetFromVector(self._contacts(context)[0])

    def _calc_count(self, context, out):
        out.SetFromVector([float(np.count_nonzero(self._contacts(context)[0] > 0.0))])

    def _calc_total(self, context, out):
        out.SetFromVector(self._contacts(context)[1])

    def _calc_strike(self, context, out):
        out.SetFromVector([1.0 if self._state.Eval(context)[6] < 0.0 else 0.0])
