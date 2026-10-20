"""Concatenates the force lists of several sources onto the plant's single port."""

from pydrake.all import AbstractValue, ExternallyAppliedSpatialForce, LeafSystem


class ForceCombiner(LeafSystem):
    """`n` abstract inputs (each a list of `ExternallyAppliedSpatialForce`), one output with all of
    them. Drake sums several forces on the same body itself."""

    def __init__(self, n: int):
        super().__init__()
        self._inputs = [
            self.DeclareAbstractInputPort(
                f"forces{i}", AbstractValue.Make([ExternallyAppliedSpatialForce()])
            )
            for i in range(n)
        ]
        self.DeclareAbstractOutputPort(
            "spatial_forces",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]),
            self._calc,
        )

    def _calc(self, context, out):
        out.set_value([f for port in self._inputs for f in port.Eval(context)])
