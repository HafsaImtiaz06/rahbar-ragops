"""query_coverage.py: matching, the exit code and the live log shape (no model needed)."""

from __future__ import annotations

import json
import math
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_coverage as cov

HERE = Path(__file__).resolve().parent
DIM = 256


def bow(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


def write(tmp: Path, golden: list[str], live: list[tuple[str, str, str]],
          threshold: float = 0.5, minimum: float = 0.85) -> list[str]:
    (tmp / "golden.yaml").write_text(yaml.safe_dump({"questions": [
        {"id": f"g{i}", "question": q} for i, q in enumerate(golden, 1)]}))
    (tmp / "live.yaml").write_text(yaml.safe_dump({"queries": [
        {"id": f"q{i}", "text": t, "kind": k, "topic": topic}
        for i, (t, k, topic) in enumerate(live, 1)]}))
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(
        {"similarity_threshold": threshold, "min_coverage": minimum}))
    return ["--live", str(tmp / "live.yaml"), "--golden", str(tmp / "golden.yaml"),
            "--config", str(tmp / "cfg.yaml"), "--out-dir", str(tmp / "out"), "--label", "t"]


GOLDEN = ["What CGPA puts a student on academic warning",
          "What is the late payment fine in the fee information"]
LIVE = [("What CGPA puts a student on academic warning please", "paraphrase", "grades"),
        ("late payment fine for fee", "paraphrase", "fees"),
        ("library opening hours saturday", "offtopic", "library"),
        ("hostel visitors rules curfew", "offtopic", "hostel")]


def test_exit_code_is_nonzero_below_the_minimum_coverage(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    args = write(tmp_path, GOLDEN, LIVE)
    assert cov.main(args, embed=bow) == 1                 # 2 of 4 covered = 50 percent < 85
    out = capsys.readouterr().out
    assert "FAIL" in out and "library (1)" in out and "hostel (1)" in out
    result = json.loads((tmp_path / "out" / "coverage-t.json").read_text())
    assert result["summary"]["covered"] == 2 and result["summary"]["coverage"] == 0.5


def test_adding_golden_questions_raises_coverage_and_passes(tmp_path: Path) -> None:
    before = cov.summarise(cov.match(cov.load_live(Path(write(tmp_path, GOLDEN, LIVE)[1])),
                                     cov.load_golden(tmp_path / "golden.yaml"), bow, 0.5),
                           0.5, 0.85)
    more = GOLDEN + ["What are the library opening hours on saturday",
                     "What are the hostel visitors rules and curfew"]
    args = write(tmp_path, more, LIVE)
    after_rows = cov.match(cov.load_live(tmp_path / "live.yaml"),
                           cov.load_golden(tmp_path / "golden.yaml"), bow, 0.5)
    after = cov.summarise(after_rows, 0.5, 0.85)
    assert after["coverage"] > before["coverage"] and after["passes"] is True
    assert cov.main(args, embed=bow) == 0


def test_cli_threshold_overrides_the_config(tmp_path: Path) -> None:
    args = write(tmp_path, GOLDEN, LIVE, threshold=0.99)
    assert cov.main(args + ["--threshold", "0.0"], embed=bow) == 0   # everything is covered at 0


def test_unreachable_embedder_exits_2(tmp_path: Path) -> None:
    def down(_texts: list[str]) -> list[list[float]]:
        raise OSError("connection refused")
    assert cov.main(write(tmp_path, GOLDEN, LIVE), embed=down) == 2


def test_calibration_suggests_a_threshold_between_the_groups(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cov.main(write(tmp_path, GOLDEN, LIVE) + ["--calibrate"], embed=bow) == 0
    out = capsys.readouterr().out
    assert "paraphrase" in out and "offtopic" in out and "suggested threshold" in out
    assert not (tmp_path / "out").exists()               # calibration writes nothing


def test_cosine_basics() -> None:
    assert cov.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cov.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cov.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_the_real_live_log_meets_the_assignment_counts() -> None:
    live = cov.load_live(HERE / "live_queries.yaml")
    kinds = Counter(q["kind"] for q in live)
    assert len(live) >= 40 and len({q["id"] for q in live}) == len(live)
    assert kinds["paraphrase"] + kinds["domain"] == 28       # about 28 in the golden domain
    assert kinds["offtopic"] >= 12                           # at least 12 from uncovered topics
    assert all(q["topic"] and q["text"].endswith("?") for q in live)
    assert cov.load_golden(HERE / "golden.yaml")
