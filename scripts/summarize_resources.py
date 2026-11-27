"""F5 step 2: turn a measurement CSV into suggested requests/limits.
Rule used (state it in the report): CPU request = avg rounded up to 50m, CPU limit = 2 x peak;
memory request = peak x 1.1, memory limit = peak x 1.5 (memory is not compressible, so limit has headroom)."""
import csv, math, re, sys, collections


def up(x, step):
    return int(math.ceil(x / step) * step)


rows = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    kind = re.sub(r"-[a-z0-9]{5,10}(-[a-z0-9]{5})?$", "", r["pod"])  # strip replicaset/job hash
    rows[kind].append((float(r["cpu_m"]), float(r["mem_mi"])))
print(f"{'workload':<22}{'samples':>8}{'cpu avg/peak (m)':>20}{'mem avg/peak (Mi)':>20}   suggested")
for k, v in rows.items():
    cpu = [c for c, _ in v]; mem = [m for _, m in v]
    ca, cp, ma, mp = sum(cpu)/len(cpu), max(cpu), sum(mem)/len(mem), max(mem)
    print(f"{k:<22}{len(v):>8}{ca:>11.0f}/{cp:<8.0f}{ma:>12.0f}/{mp:<7.0f}   "
          f"requests cpu={up(ca,50)}m mem={up(mp*1.1,32)}Mi | limits cpu={up(cp*2,100)}m mem={up(mp*1.5,64)}Mi")
