import json
from pathlib import Path

import pytest

import gate

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "gate"
GOOD = HERE / "gate.yaml"


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _run(results, baseline="baseline.json", th=None, **kw):
    return gate.evaluate(_load(results), _load(baseline), th or gate.load_thresholds(GOOD),
                         allow_fixture=True, **kw)


def test_pass_fixture_passes():
    failures, checks = _run("results_pass.json")
    assert failures == []
    assert len(checks) == 5


def test_fail_fixture_fails_on_every_rule():
    failures, _ = _run("results_fail.json")
    text = " ".join(failures)
    for word in ("recall@5", "abstention", "faithfulness", "p95", ):
        assert word in text


def test_recall_drop_boundary_is_two_points():
    res = _load("results_pass.json")
    base = _load("baseline.json")
    th = gate.load_thresholds(GOOD)
    res["metrics"]["retrieval_api"]["recall_at_5"] = 0.78      # exactly 2 pp below 0.80
    assert gate.evaluate(res, base, th, allow_fixture=True)[0] == []
    res["metrics"]["retrieval_api"]["recall_at_5"] = 0.779     # just over 2 pp
    assert any("recall@5" in f for f in gate.evaluate(res, base, th, allow_fixture=True)[0])


def test_changing_a_threshold_changes_the_result_for_the_same_results():
    th = gate.load_thresholds(GOOD)
    assert _run("results_pass.json", th=th)[0] == []
    th.min_faithfulness = 0.95
    assert any("faithfulness" in f for f in _run("results_pass.json", th=th)[0])
    th.min_faithfulness = 0.80
    th.max_p95_latency_seconds = 60
    assert any("p95" in f for f in _run("results_pass.json", th=th)[0])


def test_abstention_worse_than_baseline_fails_but_tolerance_allows():
    res = _load("results_pass.json")
    base = _load("baseline.json")
    res["metrics"]["answer"]["abstention_correctness"] = 0.9
    th = gate.load_thresholds(GOOD)
    assert any("abstention" in f for f in gate.evaluate(res, base, th, allow_fixture=True)[0])
    th.abstention_tolerance = 0.15
    assert gate.evaluate(res, base, th, allow_fixture=True)[0] == []


def test_missing_metric_fails_closed():
    res = _load("results_pass.json")
    del res["metrics"]["answer"]["faithfulness"]
    failures, _ = gate.evaluate(res, _load("baseline.json"), gate.load_thresholds(GOOD),
                                allow_fixture=True)
    assert any("missing metric answer.faithfulness" in f for f in failures)


def test_incomplete_smoke_and_fixture_are_refused():
    th = gate.load_thresholds(GOOD)
    base = _load("baseline.json")
    res = _load("results_pass.json")
    res["complete"] = False
    with pytest.raises(gate.GateError, match="not complete"):
        gate.evaluate(res, base, th, allow_fixture=True)
    res = _load("results_pass.json")
    res["profile"] = "smoke"
    with pytest.raises(gate.GateError, match="not a real evaluation"):
        gate.evaluate(res, base, th, allow_fixture=True)
    with pytest.raises(gate.GateError, match="fixture"):
        gate.evaluate(_load("results_pass.json"), base, th)


def test_missing_baseline_is_undecided_not_pass():
    with pytest.raises(gate.GateError, match="no baseline"):
        gate.evaluate(_load("results_pass.json"), None, gate.load_thresholds(GOOD),
                      allow_fixture=True)


def test_legacy_baseline_format_is_understood():
    legacy = {"recall_at_5": 0.80, "abstention_correctness": 1.0, "citation_validity": 1.0,
              "latency_seconds_p95": 100.0, "per_question": []}
    failures, _ = gate.evaluate(_load("results_pass.json"), legacy,
                                gate.load_thresholds(GOOD), allow_fixture=True)
    assert failures == []


def test_unknown_setting_in_config_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("max_p95: 5\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="unknown"):
        gate.load_thresholds(path)


def _cli(*args):
    return gate.main(["--config", str(GOOD), "--baseline", str(FIX / "baseline.json"),
                      "--allow-fixture", *args])


def test_exit_codes():
    assert _cli("--results", str(FIX / "results_pass.json")) == 0
    assert _cli("--results", str(FIX / "results_fail.json")) == 1
    assert gate.main(["--config", str(GOOD), "--baseline", str(FIX / "baseline.json"),
                      "--results", str(FIX / "results_pass.json")]) == 2   # fixture not allowed


def test_gate_config_env_var_changes_result(tmp_path, monkeypatch):
    strict = tmp_path / "strict.yaml"
    strict.write_text("min_faithfulness: 0.99\n", encoding="utf-8")
    args = ["--baseline", str(FIX / "baseline.json"), "--allow-fixture",
            "--results", str(FIX / "results_pass.json")]
    monkeypatch.setenv("GATE_CONFIG", str(GOOD))
    assert gate.main(args) == 0
    monkeypatch.setenv("GATE_CONFIG", str(strict))
    assert gate.main(args) == 1


def test_promote_refuses_fixture_and_incomplete(tmp_path):
    with pytest.raises(gate.GateError):
        gate.promote(FIX / "results_pass.json", tmp_path / "b.json")
    real = {"profile": "full", "complete": False, "metrics": {}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(real), encoding="utf-8")
    with pytest.raises(gate.GateError):
        gate.promote(path, tmp_path / "b.json")
