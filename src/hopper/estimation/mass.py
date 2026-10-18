"""Propellant dead-reckoning from the controller's own throttle commands.

Mass is not an EKF state: the mass flow is thrust / (isp g0), and thrust follows the command through
a first-order lag, so integrating the (lagged) command is accurate to a fraction of a percent of the
propellant used. Command delay is not modelled; it shifts the burn by a few steps only.
"""

import math

G0 = 9.80665


class PropellantEstimator:
    def __init__(self, params, *, prop_mass: float | None = None) -> None:
        """`params` is an `EkfParams`; `prop_mass` overrides the initial propellant mass."""
        self.p = params
        self._prop0 = params.prop_mass_initial if prop_mass is None else prop_mass
        self.reset()

    def reset(self) -> None:
        self.prop = self._prop0
        self.thrust_frac = 0.0  # modelled actual thrust fraction (throttle lag state)
        self._t: float | None = None

    @property
    def mass(self) -> float:
        return self.p.dry_mass + self.prop

    def update(self, t: float, throttle_cmd: float) -> float:
        """Advance to `t` using the throttle command that was in force over the interval."""
        p = self.p
        dt = 0.0 if self._t is None else t - self._t
        self._t = t
        if dt <= 0.0 or self.prop <= 0.0:
            return self.mass
        # 0 = engine off; any other request is clamped into the throttleable range
        target = 0.0 if throttle_cmd <= 0.0 else min(max(throttle_cmd, p.thrust_min_frac), 1.0)
        a = 1.0 - math.exp(-dt / p.throttle_tau)
        new = self.thrust_frac + a * (target - self.thrust_frac)
        thrust = p.thrust_max * 0.5 * (self.thrust_frac + new)
        self.thrust_frac = new
        self.prop = max(self.prop - thrust * dt / (p.isp * G0), 0.0)
        return self.mass
