"""compose@15: replication of the screen assignments. compose@13 and compose@14 re-tested the word, pair and triple covers
on fresh pairs (15% failed). The largest set of automatic assignments left is the auto-token@2 screen's: a pole whose
auto test held and whose trigger explains >= half of its strong activations. This round draws that auto test again.

  python -m bonsai_interp.compose.round15 protocol | select | build | score | report   (same stages and flags as round7.py)

Selection: every pole assignment-grade by the screen (atlas: test held, a trigger's coverage >= 0.5) and by nothing
compose@13 or compose@14 re-tested. Per pole, the held auto test (context_tier._prereg_layer) whose trigger has the
highest stage-1 coverage: its trigger, bin and rule.
Test: the same construction as the auto test, on the same testing windows (16400..), with a new seed and leaving out every
window the original test used (the original draw is rebuilt from its seed and checked against the original tests file):
  - fire: 8 occurrences of the trigger where the rule holds (lexical: any);
  - near misses: lexical, the 8 fire snippets with the trigger swapped for a same-shape word; other bins, 4 occurrences
    where the rule fails and 4 swaps (8 rule failures when no swap word exists);
  - placebo: 8 more fire snippets against 8 others.
The scorer and judge are compose@7's (quick-8+8@2, the last token, three preambles): the auto test's own judge.
Verdict per pole: replicated (held), unreplicated (failed), untestable (fewer than 8 fresh fire snippets or near
misses). Assignment does not change: the verdict qualifies an assignment.
"""
import datetime as dt
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from bonsai_interp import atlas
from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round7 as R7
from bonsai_interp.compose import round13 as R13
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR

SEL = {"source": "auto-token@2 screen", "assign_coverage": 0.5, "seed": 15, "exclude_rounds": [13, 14]}
EXCLUDED = "every pole not assignment-grade by the auto-token@2 screen, and every pole compose@13 or compose@14 re-tested"


def _install():
    R7.PROTOCOL, R7.BASE = "compose@15", "bonsai1.compose15"
    R7.select, R7.build, R7.report = select, build, report
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.extend([Path(ct.__file__), Path(R13.__file__), Path(__file__)]))


def _screen_assigned():
    """{(L, n, sign)} assigned by the screen test (pilot.characterized's rule for the atlas test)."""
    out = set()
    for L in range(36):
        t = atlas.load(L)
        if t is None:
            continue
        d = t.select(["neuron", "sign", "test", "hit_coverage", "hit_trigger"]).to_pydict()
        for i, n in enumerate(d["neuron"]):
            if d["test"][i] == "held":
                cov = max([c for c, tg in zip(d["hit_coverage"][i] or [], d["hit_trigger"][i] or []) if tg] or [0])
                if cov >= SEL["assign_coverage"]:
                    out.add((L, n, d["sign"][i]))
    return out


def select(a):
    P, _ = R3._setup(a)
    t0 = time.time()
    asg = R3._assigned()
    done = {R13._pole(r["neuron"]) for k in SEL["exclude_rounds"] for r in ct._read_gz(R3.OUT / f"bonsai1.compose{k}.coverage.jsonl.gz")}
    want = {k for k in _screen_assigned() if k in asg and k not in done}
    hyps = []
    for L in sorted({k[0] for k in want}):
        Pl = ct._paths(L)
        cov = {(r["neuron"], r["trigger"]): r for r in ct._read_gz(Pl["screen"])}
        best = {}
        for r in ct._read_gz(Pl["results"]):
            if r.get("outcome") != "held" or R13._pole(r["neuron"]) not in want:
                continue
            s = cov[(r["neuron"], r["trigger"])]
            if r["neuron"] not in best or s["stage1"]["coverage"] > best[r["neuron"]]["coverage"]:
                best[r["neuron"]] = {"neuron": r["neuron"], "trigger": r["trigger"], "bin": s["bin"], "rule": s.get("rule"),
                                     "coverage": s["stage1"]["coverage"], "precision": s["stage1"]["precision"],
                                     "strong_rate": s["strong_rate"]}
        hyps += best.values()
    hyps.sort(key=lambda h: GR._neuron(h["neuron"]))
    kept = hyps[::max(1, len(hyps) // a.max)][:a.max] if a.max else hyps
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "n_wanted": len(want),
                                "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(want)} screen-assigned poles, {len(hyps)} with a held auto test, kept {len(kept)}", flush=True)


def _draw(r, rng, w, p, m, toks, subs, tok, skip=frozenset()):
    """context_tier._prereg_layer's draw for one test (fire, near misses), leaving out the windows in skip:
    (fire, quiet, kinds, windows used)."""
    R = ct.RULE

    def pick(mask, k, more=()):
        got, seen = [], set(skip) | set(more)
        for j in rng.permutation(np.flatnonzero(mask)):
            if w[j] not in seen:
                seen.add(w[j]); got.append(j)
            if len(got) == k:
                break
        return got
    snip = lambda j: toks[w[j], p[j] - R["snippet"] + 1: p[j] + 1].astype(np.int64)
    fire_j = pick(m, R["n"])
    sub = subs(r["trigger"], rng)
    fire, quiet, kinds = [tok.decode(snip(j)) for j in fire_j], [], []
    swap = lambda j: tok.decode(np.concatenate([snip(j)[:-1], [sub]]))
    used = {int(w[j]) for j in fire_j}
    if r["bin"] == "lexical":
        if sub is not None:
            quiet, kinds = [swap(j) for j in fire_j], ["swap-trigger"] * len(fire_j)
    else:
        k_fail = R["n_rule_fail"] if sub is not None else R["n"]
        fail_j = pick(~m, k_fail)
        used |= {int(w[j]) for j in fail_j}
        quiet = [tok.decode(snip(j)) for j in fail_j]
        kinds = ["rule-fails"] * len(fail_j)
        if sub is not None:
            quiet += [swap(j) for j in fire_j[:R["n"] - k_fail]]
            kinds += ["swap-trigger"] * (R["n"] - k_fail)
    return fire, quiet, kinds, used, pick


def build(a):
    P, _ = R3._setup(a)
    from bonsai_interp import explore
    R = ct.RULE
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    vb, tok = ct.vocab(), explore.tokenizer()
    toks, dom, _ = ct._windows(ct.SCREEN if a.smoke else ct.TEST, a)  # smoke: the screening windows, never the test ones
    rows, n_poles, checked, t0 = [], 0, [0, 0], time.time()
    by_layer = {}
    for h in sel:
        by_layer.setdefault(GR._neuron(h["neuron"])[0], []).append(h)
    for L, hs in sorted(by_layer.items()):
        Pl = ct._paths(L)
        orig = {(r["neuron"], r["trigger"]): r for r in ct._read_gz(Pl["tests"]) if r.get("class") == "tested"}
        occ = ct._occurrences(toks, sorted({h["trigger"] for h in hs}), 4096, L, min_pos=max(R["snippet"], R["min_pos"]))
        d = np.load(Pl["hits"]) if Pl["hits"].exists() else None
        subs = ct.Substitutes(d["freq"] if d is not None else np.bincount(toks.ravel().astype(np.int64), minlength=vb["V"]), vb)
        feats = {}
        for h in hs:
            n = int(h["neuron"].split("/N")[1][:-1])
            w, p = occ.get(h["trigger"], (np.zeros(0, int), np.zeros(0, int)))
            if h["trigger"] not in feats:
                fs = [ct.features(toks[wi], int(pi), dom[wi], vb, ct.bigram_table()) for wi, pi in zip(w, p)]
                feats[h["trigger"]] = {f: np.array([x[f] for x in fs], dtype=object) for f in ct.FEATURES}
            m = ct.rule_mask(feats[h["trigger"]], h["rule"]) if h["bin"] != "lexical" and len(w) else np.ones(len(w), bool)
            skip = set()
            if not a.smoke:  # rebuild the original draw, check it, and leave its windows out
                f0, q0, _, skip, _ = _draw(h, np.random.default_rng([L, h["trigger"], n, 2]), w, p, m, toks, subs, tok)
                o = orig.get((h["neuron"], h["trigger"]))
                checked[0] += 1
                checked[1] += bool(o) and o["fire"] == f0 and o["quiet"] == q0
            rng = np.random.default_rng([L, h["trigger"], n, 2, SEL["seed"]])
            fire, quiet, kinds, used, pick = _draw(h, rng, w, p, m, toks, subs, tok, skip)
            key = f"replicate|{h['neuron']}|{h['trigger']}"
            item = {"type": "items", "kind": R7.KIND, "condition": key,
                    "label": "compose (replication of the screen): " + ct.label(h, vb)}
            if len(fire) < R["n"] or len(quiet) < R["n"]:
                rows.append({**item, "class": "untestable", "why": f"{len(fire)} fire, {len(quiet)} near misses fresh"})
                continue
            rows.append({**item, "class": "tested", "fire": fire, "quiet": quiet, "quiet_kind": kinds})
            snip = lambda j: tok.decode(toks[w[j], p[j] - R["snippet"] + 1: p[j] + 1].astype(np.int64))
            pf = pick(m, 2 * R["n"], used)  # neither the original's windows nor this draw's
            if len(pf) == 2 * R["n"]:
                rows.append({"type": "items", "kind": R7.KIND, "condition": "placebo:" + key, "label": "placebo", "class": "placebo",
                             "fire": [snip(j) for j in pf[:R["n"]]], "quiet": [snip(j) for j in pf[R["n"]:]],
                             "quiet_kind": ["placebo"] * R["n"]})
            n_poles += 1
            rows.append({"type": "test", "neuron": h["neuron"], "kind": R7.KIND, "condition": key,
                         "select": {k: h[k] for k in ("trigger", "bin", "coverage", "precision", "strong_rate")}})
        print(f"layer {L}: {len(hs)} poles ({time.time() - t0:.0f} s)", flush=True)
    sha = ct._write_gz(P["tests"], rows)
    counts = {"poles": len(sel), "tested": n_poles, "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "placebo": sum(r["type"] == "items" and r["class"] == "placebo" for r in rows),
              "original_draws_rebuilt": checked[1], "original_draws_checked": checked[0]}
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
        for k in ("all", "band " + band[GR._neuron(r["neuron"])[0]], "bin " + s["bin"],
                  "precision " + R13._bin(s["precision"], [0.3, 0.5, 0.7], ["<0.3", "0.3-0.5", "0.5-0.7", ">=0.7"]),
                  "coverage " + R13._bin(s["coverage"], [0.7, 0.9], ["0.5-0.7", "0.7-0.9", ">=0.9"])):
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
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED, "rule": ct.RULE,
                    "verdict": "replicated: the fresh auto test held; unreplicated: it failed",
                    "assignment": "unchanged: the verdict qualifies an assignment, it does not assign or unassign"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
