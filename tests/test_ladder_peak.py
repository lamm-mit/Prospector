"""The checkpoint pick rule, `scripts/analysis/ladder_peak.py`, on synthetic run dirs.

Checkpoints whose mean `gated_acc` is within the band of the best mean are
tied; a tie goes to the checkpoint with more seeds, then to the later one.
"""
import json
import subprocess
import sys

from conftest import ROOT, script

PEAK = script("ladder_peak.py")


def _run(runs, name, ck, seed, scores, cap=1024, summary=True):
    d = runs / f"{name}-ck{ck}-s{seed}--bench"
    d.mkdir(parents=True)
    (d / "results.jsonl").write_text("".join(json.dumps({"gated_acc": s}) + "\n" for s in scores))
    if summary:
        (d / "summary.json").write_text(json.dumps({"policy": {"turn_max_tokens": cap}}))


def _peak(runs, *args):
    out = subprocess.run([sys.executable, str(PEAK), "r1eval", "--benchmark", "bench", "--runs", str(runs), *args],
                         capture_output=True, text=True, cwd=ROOT)
    return out.returncode, out.stdout.strip(), out.stderr


def _grid(runs):
    """ck100 on eight seeds at 0.70, ck120 on two seeds at 0.72 and ck60 on
    three seeds at 0.50."""
    for s in range(8):
        _run(runs, "r1eval", 100, s, [0.7])
    for s in range(2):
        _run(runs, "r1eval", 120, s, [0.72])
    for s in range(3):
        _run(runs, "r1eval", 60, s, [0.5])


def test_a_tie_goes_to_more_seeds_not_to_the_higher_mean(tmp_path):
    _grid(tmp_path)
    rc, out, err = _peak(tmp_path, "--pick")
    assert rc == 0 and out == "100"
    assert "ck 120" in err and "<- tied" in err          # the table goes to stderr


def test_outside_the_band_the_best_mean_wins(tmp_path):
    _grid(tmp_path)
    assert _peak(tmp_path, "--pick", "--band", "0.01")[1] == "120"


def test_equal_seed_counts_tie_to_the_later_checkpoint(tmp_path):
    for ck, g in ((80, 0.80), (100, 0.79), (120, 0.78)):
        for s in range(3):
            _run(tmp_path, "r1eval", ck, s, [g])
    assert _peak(tmp_path, "--pick")[1] == "120"


def test_top_lists_the_best_means_first(tmp_path):
    _grid(tmp_path)
    assert _peak(tmp_path, "--top", "2")[1] == "120 100"


def test_cap_keeps_only_seeds_read_at_that_budget(tmp_path):
    _grid(tmp_path)
    for s in range(3, 8):
        _run(tmp_path, "r1eval", 120, s, [0.9], cap=512)
    assert _peak(tmp_path, "--pick")[1] == "120"           # every seed counts: 0.85 on seven
    assert _peak(tmp_path, "--pick", "--cap", "1024")[1] == "100"


def test_a_run_without_its_summary_is_not_a_seed(tmp_path):
    _grid(tmp_path)
    for s in range(2, 10):
        _run(tmp_path, "r1eval", 120, s, [0.72], summary=False)
    assert _peak(tmp_path, "--pick")[1] == "100"


def test_no_reads_exits_1_with_nothing_on_stdout(tmp_path):
    rc, out, _ = _peak(tmp_path, "--pick")
    assert rc == 1 and out == ""
