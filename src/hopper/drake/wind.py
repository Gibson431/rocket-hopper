"""Mean wind plus a first-order Gauss-Markov gust, as a Drake LeafSystem.

Same exact discretisation as `hopper.dynamics.environment.Wind.advance`. The gust is discrete state
advanced by a periodic update, so it is a function of time only: RK stages and
output callbacks never draw random numbers (docs/DRAKE_DESIGN.md Q7).
"""

import numpy as np
from pydrake.all import BasicVector, LeafSystem


class WindSource(LeafSystem):
    def __init__(self, mean_i, gust_sigma: float, gust_tau: float, seed: int, dt: float = 1e-3):
        super().__init__()
        self._mean = np.asarray(mean_i, dtype=float)
        self._sigma = gust_sigma
        self._rng = np.random.default_rng(seed)
        self._phi = np.exp(-dt / gust_tau)
        self._gain = gust_sigma * np.sqrt(1.0 - self._phi**2)
        self._gust_index = self.DeclareDiscreteState(BasicVector(np.zeros(3)))
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._advance)
        self._port = self.DeclareVectorOutputPort("wind", 3, self._calc_wind)

    def get_wind_output_port(self):
        return self._port

    def _advance(self, context, discrete_state):
        if self._sigma == 0.0:
            return
        gust = context.get_discrete_state_vector().CopyToVector()
        noise = self._rng.standard_normal(3)
        discrete_state.get_mutable_vector(self._gust_index).SetFromVector(
            self._phi * gust + self._gain * noise
        )

    def _calc_wind(self, context, out):
        out.SetFromVector(self._mean + context.get_discrete_state_vector().CopyToVector())
