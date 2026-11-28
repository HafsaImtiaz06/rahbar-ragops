#!/usr/bin/env python3
"""F3: derive the CANDIDATE indexer Job and the CANDIDATE eval Job from the real manifests.

    kubectl kustomize k8s/overlays/dev | python scripts/make_candidate_jobs.py indexer \
        --collection cand --env CHUNK_SIZE=200 --env CHUNK_OVERLAP=30 > k8s/candidate/indexer-candidate.yaml
    python scripts/make_candidate_jobs.py eval --input k8s/eval-job.yaml \
        --target http://api-candidate:8000 --env CHUNK_SIZE=200 --env CHUNK_OVERLAP=30 \
        > k8s/candidate/eval-candidate.yaml

indexer: Job `indexer` -> `indexer-candidate`. Same image, labels (app: indexer, so the existing
         NetworkPolicies still apply), volumes and securityContext. Writes ONLY the collection
         family `<collection>-<id>` + pointer `<collection>-active`. The collection base name must NOT
         start with the live base + "-" (e.g. `rahbar-v2`): the live indexer deletes every collection
         whose name starts with `rahbar-` that is not active/previous. `cand` is safe.
eval:    Job `eval` (M2) -> `eval-candidate`. Same image, thresholds ConfigMap, label app: eval (so the
         eval-egress / api-from-eval policies match). Only the Job document is emitted (the ConfigMap, PVC
         and NetworkPolicies of eval-job.yaml already exist). `--api http://api:8000` is replaced by
         --target, and results go to /results/candidate so baseline and candidate runs are never mixed.
Needs: pip install pyyaml
"""
import argparse
import copy
import sys

import yaml


def load(path: str) -> list[dict]:
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    return [d for d in yaml.safe_load_all(stream) if d]


def find_job(docs: list[dict], name: str) -> dict:
    job = next((d for d in docs if d["kind"] == "Job" and d["metadata"]["name"] == name), None)
    if job is None:
        sys.exit(f"Job '{name}' not found (found: {[(d['kind'], d['metadata']['name']) for d in docs]})")
    return copy.deepcopy(job)


def set_env(container: dict, pairs: list[str]) -> None:
    env = {e["name"]: e for e in container.get("env", [])}
    for kv in pairs:
        k, _, v = kv.partition("=")
        env[k] = {"name": k, "value": v}   # explicit env beats envFrom (the ConfigMap)
    container["env"] = list(env.values())


def clean(obj: dict) -> dict:
    obj.pop("status", None)
    for k in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        obj["metadata"].pop(k, None)
    return obj


def make_indexer(a: argparse.Namespace) -> dict:
    job = find_job(load(a.input), a.live)
    job["metadata"]["name"] = a.name or f"{a.live}-candidate"
    job["metadata"].setdefault("labels", {})["track"] = "candidate"
    job["spec"].pop("selector", None)
    tpl = job["spec"]["template"]
    tpl["metadata"].setdefault("labels", {})["track"] = "candidate"
    tpl["metadata"]["labels"].pop("controller-uid", None)
    main = next(c for c in tpl["spec"]["containers"] if c["name"] == "indexer")
    set_env(main, [f"CHROMA_COLLECTION={a.collection}", *a.env])
    return clean(job)


def make_eval(a: argparse.Namespace) -> dict:
    job = find_job(load(a.input), a.live)
    job["metadata"]["name"] = a.name or f"{a.live}-candidate"
    job["metadata"].setdefault("labels", {})["track"] = "candidate"
    tpl = job["spec"]["template"]
    tpl["metadata"].setdefault("labels", {})["track"] = "candidate"
    main = next(c for c in tpl["spec"]["containers"] if c["name"] == "eval")
    script = main["args"][0]
    for old, new in (("http://api:8000", a.target),
                     ("--out-dir /results", "--out-dir /results/candidate"),
                     ("--results-dir /results", "--results-dir /results/candidate")):
        if old not in script:
            sys.exit(f"expected '{old}' in the eval Job script; k8s/eval-job.yaml changed, update this tool")
        script = script.replace(old, new)
    main["args"] = [script]
    set_env(main, a.env)
    return clean(job)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    i = sub.add_parser("indexer")
    i.add_argument("--input", default="-")
    i.add_argument("--live", default="indexer")
    i.add_argument("--name")
    i.add_argument("--collection", required=True)
    i.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    e = sub.add_parser("eval")
    e.add_argument("--input", default="k8s/eval-job.yaml")
    e.add_argument("--live", default="eval")
    e.add_argument("--name")
    e.add_argument("--target", required=True)
    e.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    a = ap.parse_args()
    job = make_indexer(a) if a.kind == "indexer" else make_eval(a)
    print(yaml.safe_dump(job, sort_keys=False))


if __name__ == "__main__":
    main()
