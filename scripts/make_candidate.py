#!/usr/bin/env python3
"""F3: derive the CANDIDATE api Deployment + Service from the real, rendered live manifests, so it can never
drift from your base (same image, probes, securityContext, NetworkPolicy-relevant labels).

    kubectl kustomize k8s/overlays/dev | python scripts/make_candidate.py \
        --env TOP_K=8 --env CHROMA_COLLECTION=rahbar-v2 > k8s/candidate/api-candidate.yaml

* name api -> api-candidate, 1 replica, no HPA; pod labels = live labels + track=candidate
  (same `app` label, so the existing NetworkPolicies still match; Service selects track=candidate only).
* --env KEY=VAL overrides the ConfigMap value for that key (explicit container env beats envFrom).
Needs: pip install pyyaml
"""
import argparse
import copy
import sys

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="-")
    ap.add_argument("--live", default="api", help="name of the live Deployment and Service")
    ap.add_argument("--name", default="api-candidate")
    ap.add_argument("--container", default=None, help="container to override env on (default: first)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL")
    a = ap.parse_args()

    docs = [d for d in yaml.safe_load_all(sys.stdin if a.input == "-" else open(a.input)) if d]
    dep = next((d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == a.live), None)
    svc = next((d for d in docs if d["kind"] == "Service" and d["metadata"]["name"] == a.live), None)
    if not dep or not svc:
        sys.exit(f"Deployment/Service named '{a.live}' not found in input (found: "
                 f"{[(d['kind'], d['metadata']['name']) for d in docs]})")

    c = copy.deepcopy(dep)
    c["metadata"]["name"] = a.name
    c["metadata"].setdefault("labels", {})["track"] = "candidate"
    c["spec"]["replicas"] = 1
    c["spec"]["selector"]["matchLabels"]["track"] = "candidate"
    tpl = c["spec"]["template"]
    tpl.setdefault("metadata", {}).setdefault("labels", {})["track"] = "candidate"
    cont = next((x for x in tpl["spec"]["containers"] if a.container in (None, x["name"])), None)
    if cont is None:
        sys.exit(f"container {a.container} not found")
    env = {e["name"]: e for e in cont.get("env", [])}
    env["RELEASE_TRACK"] = {"name": "RELEASE_TRACK", "value": "candidate"}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = {"name": k, "value": v}
    cont["env"] = list(env.values())

    s = copy.deepcopy(svc)
    s["metadata"]["name"] = a.name
    s["metadata"].setdefault("labels", {})["track"] = "candidate"
    s["spec"]["selector"] = {**s["spec"]["selector"], "track": "candidate"}
    for k in ("clusterIP", "clusterIPs"):
        s["spec"].pop(k, None)
    for o in (c, s):
        o.pop("status", None)
        for k in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
            o["metadata"].pop(k, None)
    print(yaml.safe_dump_all([c, s], sort_keys=False))


if __name__ == "__main__":
    main()
