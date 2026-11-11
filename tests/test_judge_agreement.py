import json
import subprocess
import sys

import pytest

from llm_judge_agreement import analyse, demo_rows, main, wilson


def test_demo_numbers():
    r = analyse(demo_rows())
    assert r["confusion"] == {"tp": 30, "fn": 10, "fp": 2, "tn": 78}
    assert round(r["kappa"], 3) == 0.763
    assert r["tpr"]["met"] is False and r["tnr"]["met"] is True and r["ok"] is False
    assert round(r["judge_only"]["corrected"], 3) == 0.759


def test_bars_can_be_set():
    assert analyse(demo_rows(), min_tpr=0.5, min_tnr=0.9)["ok"] is True


def test_wilson_bounds():
    low, high = wilson(0, 10)
    assert low < 1e-9 and 0 < high < 0.35


@pytest.mark.parametrize("rows", [
    [{"case_id": "a", "human_label": "pass", "judge_label": "maybe"}],
    [{"case_id": "a", "human_label": "pass", "judge_label": "pass"}, {"case_id": "a", "human_label": "fail", "judge_label": "fail"}],
    [{"case_id": "a", "human_label": "pass", "judge_label": "pass"}],
])
def test_bad_input(rows):
    with pytest.raises(ValueError):
        analyse(rows)


def test_cli_exit_codes(tmp_path, capsys):
    assert main(["--demo"]) == 1
    assert main(["--demo", "--min-tpr", "0.5"]) == 0
    bad = tmp_path / "bad.csv"
    bad.write_text("case_id,human_label,judge_label\na,pass,maybe\n")
    assert main([str(bad)]) == 2


def test_cli_json(capsys):
    main(["--demo", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["labelled"] == 120


def test_module_runs():
    out = subprocess.run([sys.executable, "-m", "llm_judge_agreement", "--demo"], capture_output=True, text=True)
    assert out.returncode == 1 and "decision:" in out.stdout
