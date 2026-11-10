import numpy as np
import pytest

from roomeq.analysis import log_grid
from roomeq.audio import SimulatedIO
from roomeq.export import autoeq_text, minidsp_biquads_text
from roomeq.optimizer import EQConfig, design_eq
from roomeq.peq import Filter, biquad, eq_db
from roomeq.sweep import (SweepParams, analyse_recording, build_stimulus, exp_sweep,
                          timing_from_sequence)
from roomeq.target import make_target


def test_peaking_filter_gain_at_fc():
    f = Filter("PK", 100.0, -6.0, 4.0)
    assert eq_db([f], np.array([100.0]), 48000)[0] == pytest.approx(-6.0, abs=0.01)


def test_shelves_reach_gain():
    lo = eq_db([Filter("LSC", 100.0, 5.0, 0.7)], np.array([10.0, 10000.0]), 48000)
    hi = eq_db([Filter("HSC", 2000.0, -4.0, 0.7)], np.array([50.0, 20000.0]), 48000)
    assert lo == pytest.approx([5.0, 0.0], abs=0.1)
    assert hi == pytest.approx([0.0, -4.0], abs=0.15)


def test_optimizer_removes_room_modes():
    grid = log_grid(10, 22000)
    target = make_target("default").shape_db(grid)
    modes = [Filter("PK", 45, 9, 6), Filter("PK", 72, 7, 8), Filter("PK", 130, 5, 4),
             Filter("PK", 1500, 3, 1.0)]
    speaker_hp = np.where(grid < 40, -24 * np.log2(40 / np.minimum(grid, 40)), 0)
    measured = target + eq_db(modes, grid, 48000) + speaker_hp
    res = design_eq(grid, measured, measured, target, EQConfig())
    after = measured + res.eq
    m = (grid >= 45) & (grid <= 10000)
    before_rms = np.sqrt(np.mean((measured - target)[m] ** 2))
    after_rms = np.sqrt(np.mean((after - target)[m] ** 2))
    assert before_rms > 2.0
    assert after_rms < 0.5
    assert len(res.filters) <= 10
    # speaker extension found; no boost below it
    assert 30 <= res.f_lo <= 45
    assert np.max(res.eq[grid < 0.8 * res.f_lo]) < 1.0


def test_dips_only_partly_filled():
    grid = log_grid(10, 22000)
    target = np.zeros_like(grid)
    null = eq_db([Filter("PK", 80, -20, 10)], grid, 48000)
    res = design_eq(grid, null, null + 3, target, EQConfig(max_boost=4, min_freq=20))
    assert np.max(res.eq) <= 4.5


def test_sweep_recovers_simulated_room_and_delay():
    p = SweepParams(duration=3.0)
    sweep = exp_sweep(p)
    io = SimulatedIO(p.fs, seed=3)
    stim, starts = build_stimulus(sweep, p, ["L", "R", "L"], {"L": 0, "R": 1})
    rec = io.play_record(stim)
    res = analyse_recording(rec, sweep, p, starts, ["L", "R", "L"])
    assert [ir.channel for ir in res.irs] == ["L", "R", "L"]
    assert all(ir.pnr_db > 50 for ir in res.irs)
    t = timing_from_sequence(res.irs, starts, p.fs)
    assert t["r_minus_l_ms"] == pytest.approx(0.6, abs=0.05)


def test_exports():
    fl = [Filter("PK", 63.5, -6.2, 5.0), Filter("HSC", 8000, -2, 0.7)]
    txt = autoeq_text(fl, -1.0)
    assert txt.splitlines()[1] == "Filter 1: ON PK Fc 63.5 Hz Gain -6.2 dB Q 5.00"
    mdsp = minidsp_biquads_text(fl)
    assert mdsp.count("biquad") == 10
    b, a = biquad(fl[0], 96000)
    assert f"a1={-a[1]:.15f}," in mdsp


# --- verify ------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim_correction(tmp_path_factory):
    from roomeq.cli import main
    out = tmp_path_factory.mktemp("corr")
    assert main(["measure", "--simulate", "--yes", "--no-open", "--positions", "2",
                 "--sweep-length", "2.5", "--out", str(out)]) == 0
    return out


@pytest.mark.parametrize("how,expect_status,expect_text", [
    ("applied", "PASS", None),
    ("none", "FAIL", "not active"),
    ("swapped", "FAIL", "SWAPPED"),
    ("halfrate", "FAIL", "HALF its frequency"),
])
def test_verify_diagnoses(sim_correction, how, expect_status, expect_text):
    import json
    from roomeq.cli import main
    rc = main(["verify", str(sim_correction), "--simulate", "--simulate-eq", how, "--yes",
               "--no-open"])
    runs = sorted(sim_correction.glob("verify_channel_*"), key=lambda p: p.stat().st_mtime)
    res = json.loads((runs[-1] / "verify.json").read_text())
    assert res["status"] == expect_status
    assert rc == (2 if expect_status == "FAIL" else 0)
    if expect_text:
        assert any(expect_text in f for f in res["findings"])
    assert (runs[-1] / "verify_report.png").exists()
    # verify never touches the correction itself
    assert (sim_correction / "EqualizerAPO.txt").exists()
