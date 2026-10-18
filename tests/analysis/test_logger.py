import pandas as pd
import pytest

from hopper.analysis.logger import Logger
from hopper.sim.events import Outcome
from hopper.types import make_state

STATE_COLS = [
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
EXTRA_COLS = [
    "mass",
    "thrust",
    "cmd_throttle",
    "cmd_gim_x",
    "cmd_gim_y",
    "cmd_roll",
    "contact_force",
    "n_contacts",
    "event",
]


def fill(logger, n, event_last=None):
    for k in range(n):
        x = make_state([k, 2 * k, 3.0], [0, 0, -k], None, [0, 0, 0.1 * k], 10.0 - 0.01 * k)
        ev = event_last if k == n - 1 else None
        logger.log(
            0.001 * k, x, 90.0 + k, 1000.0 + k, (0.5, (0.01, -0.02), 0.25), 7.5 * k, k % 5, ev
        )


def test_dataframe_has_the_documented_columns_in_order():
    lg = Logger()
    fill(lg, 3)
    assert list(lg.to_dataframe().columns) == ["t", *STATE_COLS, *EXTRA_COLS]


def test_values_land_in_the_right_columns():
    lg = Logger()
    fill(lg, 4)
    df = lg.to_dataframe()
    assert len(df) == 4
    assert df["t"].iloc[3] == pytest.approx(0.003)
    assert df["pos_y"].iloc[3] == 6.0
    assert df["vel_z"].iloc[2] == -2.0
    assert df["omega_z"].iloc[3] == pytest.approx(0.3)
    assert df["quat_w"].iloc[0] == 1.0
    assert df["prop"].iloc[1] == pytest.approx(9.99)
    assert df["cmd_gim_y"].iloc[0] == -0.02
    assert df["cmd_roll"].iloc[0] == 0.25
    assert df["contact_force"].iloc[2] == 15.0
    assert df["n_contacts"].iloc[3] == 3
    assert df["mass"].iloc[1] == 91.0


def test_event_is_empty_until_the_terminal_row():
    lg = Logger()
    fill(lg, 3, event_last=Outcome.CRASHED)
    ev = lg.to_dataframe()["event"].tolist()
    assert ev == ["", "", "CRASHED"]


def test_decimation_keeps_every_nth_row_and_always_the_terminal_row():
    lg = Logger(decimation=4)
    fill(lg, 10, event_last=Outcome.LANDED)
    df = lg.to_dataframe()
    assert df["t"].tolist() == pytest.approx([0.0, 0.004, 0.008, 0.009])
    assert df["event"].iloc[-1] == "LANDED"


def test_grows_past_initial_capacity():
    lg = Logger()
    fill(lg, 5000)
    df = lg.to_dataframe()
    assert len(df) == 5000
    assert df["pos_x"].iloc[-1] == 4999.0


def test_invalid_decimation_rejected():
    with pytest.raises(ValueError):
        Logger(decimation=0)


def test_parquet_round_trip_preserves_values_and_metadata(tmp_path):
    lg = Logger(metadata={"seed": 7, "config_hash": "abc123"})
    fill(lg, 50, event_last=Outcome.TIMEOUT)
    path = tmp_path / "run.parquet"
    lg.save(path)
    df = Logger.load(path)
    pd.testing.assert_frame_equal(df, lg.to_dataframe(), check_exact=True)
    assert df.attrs["seed"] == 7
    assert df.attrs["config_hash"] == "abc123"
