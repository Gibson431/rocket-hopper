"""Run log: pre-allocated arrays, parquet round trip with run metadata."""

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hopper.sim.events import Outcome
from hopper.types import N_STATE, Command

STATE_COLUMNS = [
    *(f"pos_{a}" for a in "xyz"),
    *(f"vel_{a}" for a in "xyz"),
    *(f"quat_{a}" for a in "wxyz"),
    *(f"omega_{a}" for a in "xyz"),
    "prop",
    "thr",
    "gim_x",
    "gim_y",
    "gimr_x",
    "gimr_y",
]
SCALAR_COLUMNS = [
    "mass",
    "thrust",
    "cmd_throttle",
    "cmd_gim_x",
    "cmd_gim_y",
    "cmd_roll",
    "contact_force",
    "n_contacts",
]
COLUMNS = ["t", *STATE_COLUMNS, *SCALAR_COLUMNS, "event"]

_META_KEY = b"hopper_meta"
_EVENT_NAMES = {o.value: o.name for o in Outcome}
_N_FLOAT = 1 + N_STATE + len(SCALAR_COLUMNS)  # t, state, scalars (n_contacts stored as float)
_INITIAL_CAPACITY = 1024


def save_frame(df: pd.DataFrame, path) -> None:
    """Write a log frame to parquet with `df.attrs` (seed, config hash) in the file metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}), _META_KEY: json.dumps(df.attrs).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)


class Logger:
    """Collects one row per physics step (or every `decimation`-th); the terminal row is kept."""

    def __init__(
        self,
        decimation: int = 1,
        metadata: dict | None = None,
        extra_columns: tuple[str, ...] = (),
    ) -> None:
        if decimation < 1:
            raise ValueError(f"decimation must be >= 1, got {decimation}")
        self.decimation = decimation
        self.metadata = dict(metadata or {})
        self.extra_columns = tuple(extra_columns)  # e.g. estimator output, logged after the scalars
        self._calls = 0
        self._n = 0
        self._data = np.empty((_INITIAL_CAPACITY, _N_FLOAT + len(self.extra_columns)))
        self._event = np.zeros(_INITIAL_CAPACITY, dtype=np.int8)

    def wants_row(self) -> bool:
        """Whether the next `log` call will be stored (unless it carries an event)."""
        return self._calls % self.decimation == 0

    def skip(self) -> None:
        """Count a step that is not stored (same effect as a `log` call that is dropped)."""
        self._calls += 1

    def log(
        self,
        t: float,
        x: np.ndarray,
        mass: float,
        thrust: float,
        cmd: tuple[float, tuple[float, float], float] | Command,
        contact_force: float,
        n_contacts: int,
        event: Outcome | None = None,
        extra: Sequence[float] | None = None,
    ) -> None:
        if len(extra if extra is not None else ()) != len(self.extra_columns):
            raise ValueError(f"expected {len(self.extra_columns)} extra values, got {extra}")
        keep = self.wants_row() or event is not None
        self._calls += 1
        if not keep:
            return
        if self._n == len(self._data):
            self._data = np.concatenate([self._data, np.empty_like(self._data)])
            self._event = np.concatenate([self._event, np.zeros_like(self._event)])
        if isinstance(cmd, Command):
            throttle, (gx, gy), roll = cmd.throttle, cmd.gimbal, cmd.roll_torque
        else:
            throttle, (gx, gy), roll = cmd
        row = self._data[self._n]
        row[0] = t
        row[1 : 1 + N_STATE] = x
        row[1 + N_STATE : _N_FLOAT] = (
            mass,
            thrust,
            throttle,
            gx,
            gy,
            roll,
            contact_force,
            n_contacts,
        )
        row[_N_FLOAT:] = () if extra is None else extra
        self._event[self._n] = 0 if event is None else event.value
        self._n += 1

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self._data[: self._n].copy(), columns=[*COLUMNS[:-1], *self.extra_columns]
        )
        df["n_contacts"] = df["n_contacts"].astype(np.int64)
        df["event"] = [_EVENT_NAMES.get(int(c), "") for c in self._event[: self._n]]
        df.attrs.update(self.metadata)
        return df

    def save(self, path) -> None:
        save_frame(self.to_dataframe(), path)

    @staticmethod
    def load(path) -> pd.DataFrame:
        """The logged frame, with the run metadata (seed, config hash) in `df.attrs`."""
        table = pq.read_table(path)
        df = table.to_pandas()
        raw = (table.schema.metadata or {}).get(_META_KEY)
        df.attrs.update(json.loads(raw) if raw else {})
        return df
