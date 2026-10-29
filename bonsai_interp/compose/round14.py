"""compose@14: replication of the pair and triple covers. compose@13 found that a word cover assigned on one 8 + 8 test
often fails a second test on fresh pairs when its selection precision is low (4.9% at >= 0.5, 61% under 0.2). This
round runs the same check on every pole assigned by a pair or triple cover (compose@10, 11, and compose@12's pair sets).

  python -m bonsai_interp.compose.round14 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py): the same test split, reader and judge (quick-8+8@2). Each cover is tested again
with its own round's test (round10._pair_battery, round11._triple_battery: the same strata), on pairs drawn with a new
seed, with a placebo. Verdict per pole: replicated (held), unreplicated (failed), untestable (fewer than 8 pairs).
Assignment does not change: the verdict qualifies an assigned cover.
"""
import datetime as dt
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round7 as R7
from bonsai_interp.compose import round10 as R10
from bonsai_interp.compose import round11 as R11
from bonsai_interp.compose import round13 as R13
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR

SEL = {"sources": [10, 11, 12], "kinds": ["pair", "triple"], "cap": 20000}
EXCLUDED = "every pole not assignment-grade by a pair or triple cover (atlas, after compose@13)"
PREFIX = "replicate:"  # a new key, so the batteries draw new pairs (their seed is the key's)


def _install():
    R7.PROTOCOL, R7.BASE = "compose@14", "bonsai1.compose14"
    R7.select, R7.build, R7.report = select, build, report
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.extend([Path(R10.__file__), Path(R11.__file__), Path(__file__)]))


def select(a):
    P, _ = R3._setup(a)
    t0 = time.time()
    asg = R3._assigned()
    hyps = []
    for r in SEL["sources"]:
        for row in ct._read_gz(R3.OUT / f"bonsai1.compose{r}.coverage.jsonl.gz"):
            pt = row["parts"]
            if len(pt) == 1 and pt[0]["kind"] in SEL["kinds"] and R13._pole(row["neuron"]) in asg:
                hyps.append({"neuron": row["neuron"], "source": f"compose@{r}", "kind": pt[0]["kind"], "tokens": pt[0]["tokens"],
                             "coverage": pt[0]["coverage"], "precision": pt[0]["precision"]})
    hyps.sort(key=lambda h: GR._neuron(h["neuron"]))
    kept = hyps[::max(1, len(hyps) // a.max)][:a.max] if a.max else hyps[:SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} assigned pair and triple covers, kept {len(kept)}", flush=True)


def _name(h):
    return PREFIX + h["kind"] + "|" + ",".join(map(str, h["tokens"]))


def build(a):
    P, _ = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    pair = {_name(h): h["tokens"] for h in sel if h["kind"] == "pair"}
    triple = {_name(h): h["tokens"] for h in sel if h["kind"] == "triple"}
    t0 = time.time()
    bat = {**(R10._pair_battery(pair, split) if pair else {}), **(R11._triple_battery(triple, split) if triple else {})}
    print(f"batteries for {len(pair)} pair and {len(triple)} triple sets in {time.time() - t0:.0f} s", flush=True)
    tk, V = explore.tokenizer(), ct.vocab()["V"]

    def items(x, kind):
        parts = (x // V, x % V) if kind == "pair" else (x // (V * V), x // V % V, x % V)
        return repr("|".join(tk.decode([int(t)]) for t in parts))
    rows, tested = [], set()
    for key in sorted(set(pair) | set(triple)):
        kind = "pair" if key in pair else "triple"
        T = (pair if kind == "pair" else triple)[key]
        txt = ", ".join(items(x, kind) for x in T)
        lab = (f"compose (replication of a {kind} cover): fires on {txt}, not on the same tokens with another "
               f"{'previous token' if kind == 'pair' else 'token two back'}, nor the same {'previous token' if kind == 'pair' else 'first two'} "
               f"before another token (fresh pairs)")
        pairs, plac = bat[key], bat["placebo:" + key]
        item = {"type": "items", "kind": R7.KIND, "condition": key, "label": lab}
        if len(pairs) < C.N_TEST:
            rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
            continue
        tested.add(key)
        rows.append({**item, "class": "tested", "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                     "quiet_kind": [s for _, _, s in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": R7.KIND, "condition": "placebo:" + key, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
    n_poles = 0
    for h in sel:
        k = _name(h)
        if k not in tested:
            continue
        n_poles += 1
        rows.append({"type": "test", "neuron": h["neuron"], "kind": R7.KIND, "condition": k,
                     "select": {"source": h["source"], "part_kind": h["kind"], "items": len(h["tokens"]),
                                "coverage": h["coverage"], "precision": h["precision"]}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "poles": n_poles, "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": R7.PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def report(a):
    P = R3._paths(a.smoke) if a.smoke else R3._paths()
    from bonsai_interp.graph.protocols import v2_stats
    res = ct._read_gz(P["results"])
    pre = json.loads(P["prereg"].read_text(encoding="utf-8")) if P["prereg"].exists() else {}
    band = {L: f"{lo}-{hi}" for lo, hi in GR.BANDS for L in range(lo, hi + 1)}
    tab, rows, plac = {}, [], [0, 0]
    for r in res:
        if r["placebo"]:
            plac[0] += 1
            plac[1] += r["outcome"] == "held"
            continue
        v = "replicated" if r["outcome"] == "held" else "unreplicated"
        s = r["select"]
        prec = s["precision"]
        for k in ("all", "band " + band[GR._neuron(r["neuron"])[0]], "source " + s["source"], "kind " + s["part_kind"],
                  "precision " + R13._bin(prec, [0.3, 0.5, 0.7], ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"])):
            e = tab.setdefault(k, {"poles": 0, "replicated": 0, "unreplicated": 0})
            e["poles"] += 1
            e[v] += 1
        st = v2_stats(r)
        rows.append({"neuron": r["neuron"], "verdict": v, "select": s, "label": r["label"], "auroc": round(st[0], 3),
                     "median_fire": round(st[1], 3)})
    ct._write_gz(P["coverage"], rows)
    tot = tab.get("all", {})
    summ = {"protocol": R7.PROTOCOL, "kind": R7.KIND, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": ct._commit(), "prereg_commit": (GR.R._alone(P["prereg"]) or "")[:12] if not a.smoke else None,
            "poles_tested": len(rows), "replicated": tot.get("replicated", 0), "unreplicated": tot.get("unreplicated", 0),
            "placebo_tests": plac[0], "placebo_held": plac[1], "breakdown": tab,
            "unreplicated_by_layer": np.bincount([GR._neuron(r["neuron"])[0] for r in rows if r["verdict"] == "unreplicated"],
                                                 minlength=36).tolist(),
            "results_file": P["results"].name, "results_sha256": hashlib.sha256(gzip.open(P["results"], "rb").read()).hexdigest(),
            "coverage_file": P["coverage"].name, "tests_sha256": pre.get("tests_sha256")}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("poles_tested", "replicated", "unreplicated", "placebo_tests", "placebo_held",
                                           "breakdown")}, indent=1))


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED,
                    "verdict": "replicated: the fresh test held; unreplicated: it failed",
                    "assignment": "unchanged: the verdict qualifies an assigned cover, it does not assign or unassign"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
