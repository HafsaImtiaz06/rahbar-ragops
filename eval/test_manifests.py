"""The F2 manifest, checked offline (no cluster): its thresholds ConfigMap must match
eval/gate.yaml, and editing that ConfigMap text must change the gate result for the same results."""

from pathlib import Path

import yaml

import gate

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures" / "gate"


def _docs():
    return list(yaml.safe_load_all((ROOT / "k8s" / "eval-job.yaml").read_text(encoding="utf-8")))


def _kind(kind):
    return next(d for d in _docs() if d["kind"] == kind and d["metadata"]["name"] in
                ("eval", "rahbar-eval-thresholds", "eval-results"))


def test_configmap_matches_gate_yaml():
    cm = _kind("ConfigMap")
    on_cluster = yaml.safe_load(cm["data"]["gate.yaml"])
    in_repo = yaml.safe_load((ROOT / "eval" / "gate.yaml").read_text(encoding="utf-8"))
    assert on_cluster == in_repo


def test_job_reads_thresholds_from_the_configmap_and_writes_to_a_volume():
    job = _kind("Job")
    spec = job["spec"]["template"]["spec"]
    container = spec["containers"][0]
    mounts = {m["name"]: m for m in container["volumeMounts"]}
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["GATE_CONFIG"] == mounts["thresholds"]["mountPath"] + "/gate.yaml"
    volumes = {v["name"]: v for v in spec["volumes"]}
    assert volumes["thresholds"]["configMap"]["name"] == "rahbar-eval-thresholds"
    assert volumes["results"]["persistentVolumeClaim"]["claimName"] == "eval-results"
    assert mounts["results"]["mountPath"] == "/results"
    assert "--out-dir /results" in container["args"][0]
    assert "gate.py" in container["args"][0]
    assert job["spec"]["backoffLimit"] == 0
    assert container["securityContext"]["readOnlyRootFilesystem"] is True


def test_the_job_is_not_in_the_base_kustomization():
    base = (ROOT / "k8s" / "base" / "kustomization.yaml").read_text(encoding="utf-8")
    assert "eval" not in base
    top = yaml.safe_load((ROOT / "k8s" / "kustomization.yaml").read_text(encoding="utf-8"))
    assert "eval-job.yaml" in top["resources"]


def test_editing_the_configmap_threshold_flips_the_gate(tmp_path):
    text = _kind("ConfigMap")["data"]["gate.yaml"]
    results = str(FIX / "results_pass.json")
    base = str(FIX / "baseline.json")

    def run(config_text):
        path = tmp_path / "gate.yaml"
        path.write_text(config_text, encoding="utf-8")
        return gate.main(["--config", str(path), "--baseline", base, "--results", results,
                          "--allow-fixture"])

    assert run(text) == 0                                   # as shipped
    stricter = text.replace("min_faithfulness: 0.80", "min_faithfulness: 0.95")
    assert stricter != text
    assert run(stricter) == 1                               # one edited value, same results
