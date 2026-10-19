"""Simulator setup: fixed-step RK3 at the physics step (docs/DRAKE_DESIGN.md Q5)."""

from pydrake.all import ResetIntegratorFromFlags, Simulator


def make_simulator(diagram, dt: float = 1e-3) -> Simulator:
    sim = Simulator(diagram)
    integrator = ResetIntegratorFromFlags(sim, "runge_kutta3", dt)
    integrator.set_fixed_step_mode(True)
    return sim
