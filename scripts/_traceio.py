"""Shared reader for the trace scripts (stdlib only; runs on your laptop, not in the API image)."""
import glob
import hashlib
import json
import os
import sys


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def iter_records(sources):
    """Yield trace dicts from files, dirs, globs or '-' (stdin, e.g. kubectl logs). Lines that are not JSON,
    or are JSON without a trace_id (the API's ordinary request logs), are skipped."""
    paths = []
    for s in sources:
        if s == "-":
            paths.append("-")
        elif os.path.isdir(s):
            paths += sorted(glob.glob(os.path.join(s, "*.jsonl")))
        else:
            paths += sorted(glob.glob(s)) or [s]
    for p in paths:
        fh = sys.stdin if p == "-" else open(p, encoding="utf-8")
        try:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "trace_id" in rec:
                    yield rec
        finally:
            if fh is not sys.stdin:
                fh.close()
