"""F5 load generator (stdlib only). Questions are read from any file with `question:` / "question": entries
(your golden yaml/jsonl), or one question per line.
    kubectl -n rahbar port-forward deploy/api 18000:8000 &
    python scripts/load_test.py --questions eval/golden.yaml --n 60 --c 4
By default repeated questions can be served by the semantic cache; pass --bust to append a unique token to each
question and measure the uncached path (do both and report both in F5)."""
import argparse, itertools, json, re, statistics, threading, time, urllib.request


def load_questions(path):
    txt = open(path, encoding="utf-8").read()
    qs = [m.group(1).strip().strip('"\'') for m in re.finditer(r'["\']?question["\']?\s*:\s*(.+)', txt)]
    qs = [q.rstrip(",").strip('"\'') for q in qs]
    return qs or [l.strip() for l in txt.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:18000/api/ask")
    ap.add_argument("--questions", default="eval/golden.yaml")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--c", type=int, default=4)
    ap.add_argument("--bust", action="store_true", help="append a unique token to every question (defeats the cache)")
    a = ap.parse_args()
    pool = itertools.cycle(load_questions(a.questions))
    lock, lat, errs, left = threading.Lock(), [], [0], [a.n]

    def worker():
        while True:
            with lock:
                if left[0] <= 0:
                    return
                left[0] -= 1
                q = next(pool)
                i = a.n - left[0]
            if a.bust:
                q = f"{q} (ref {i})"
            req = urllib.request.Request(a.url, json.dumps({"question": q}).encode(), {"Content-Type": "application/json"})
            t = time.perf_counter()
            try:
                urllib.request.urlopen(req, timeout=180).read()
                with lock:
                    lat.append((time.perf_counter() - t) * 1000)
            except Exception:
                with lock:
                    errs[0] += 1

    ts = [threading.Thread(target=worker) for _ in range(a.c)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lat.sort()
    if lat:
        print(f"ok={len(lat)} errors={errs[0]} p50={statistics.median(lat):.0f}ms p95={lat[int(.95 * (len(lat) - 1))]:.0f}ms")
    else:
        print(f"no successful requests, errors={errs[0]}")


if __name__ == "__main__":
    main()
