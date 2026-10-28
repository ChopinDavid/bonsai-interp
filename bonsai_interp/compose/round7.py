"""compose@7: union covers. A pole that fires for two unrelated reasons has no single hypothesis covering half of it;
a label of two parts can ("X where c holds, or Y"). Each part is tested on its own, and the pole is assignment-grade
only when every part holds and together they cover at least half of the pole (the survey's 50% rule).

  python -m bonsai_interp.compose.round7 protocol | select | build | score | report   (as round3.py)

The engine is compose@3's (round3.py): the same event sample (pinned by hash), the test split, the reader and judge.
The library is compose@6's (conds4.py: read-token, format, shifted and next-token conditions). A part is one of:
  word   a token set T (at most 8 tokens) alone; tested as a token of T against other tokens of the same corpus-frequency
         band and domain
  conj   a token of T where condition c holds, c mattering within T (compose@3's rule); tested as the same token with c
         on against c off
Selection, per eligible pole not assignment-grade and not tested by compose@3-6, with >= 32 sampled activations in at
least 8 windows:
  - every part needs coverage >= PART["min_cov"] of the activations it is chosen on, and precision >= 0.1;
  - part 1 is the candidate with the highest coverage; part 2 is chosen on the activations part 1 leaves;
  - a single part covering half the pole is the hypothesis alone; otherwise two parts that share no token (distinct
    triggers) form a union;
  - at most one part is a word part (two word parts are one word set, such as a word and its inflection); a word part
    takes the fewest of the pole's commonest tokens (at most 8) that reach the part's coverage floor, or half the pole;
  - the union must cover >= 0.5 of the pole (a single part covering >= 0.5 alone is a one-part hypothesis);
  - hypotheses are ranked by union coverage x the smaller precision; the best SEL["cap"][1] one-part and
    SEL["cap"][2] two-part hypotheses are tested (one-part covers rank higher, so the two kinds are capped apart).
"""
import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import conds4 as CD
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round5 as R5
from bonsai_interp.events import sample as EV
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR
from bonsai_interp.sweeplib import kinds as SK

PROTOCOL, KIND, BASE = "compose@7", "cover", "bonsai1.compose7"
SEL = {"min_events": 32, "min_windows": 8, "union_cov": 0.5, "min_precision": 0.1, "min_lift_in_tokens": 1.5,
       "token_share": 0.8, "max_tokens": 8, "candidates": 15, "cap": {1: 4000, 2: 2000}}
PART = {"min_cov": 0.2}


def _install():
    R3.PROTOCOL, R3.BASE, R3.KIND = PROTOCOL, BASE, KIND
    R3.CD = CD
    R3.CODE = [Path(R3.__file__), Path(CD.__file__), Path(CD.C2.__file__), Path(CD.C3.__file__), Path(EV.__file__),
               Path(R5.__file__), Path(__file__)]


def _excluded():
    return R3._assigned() | R5._tested(3, 4, 5, 6)


def _parts(cm, tk, rem, count, hist, alltok, nm):
    """Candidate parts on the activations rem leaves: [(mask over all events, precision, kind, condition, tokens)]."""
    out = []
    vals, cs = np.unique(tk[rem], return_counts=True)
    o = np.argsort(-cs, kind="stable")
    for goal in (PART["min_cov"], SEL["union_cov"]):  # word parts: the fewest commonest tokens reaching each goal
        for k in range(1, SEL["max_tokens"] + 1):
            T = vals[o[:k]]
            m = np.isin(tk, T)
            if (m & rem).mean() >= goal:
                prec = m.mean() * count / max(int(alltok[T].sum()), 1)
                if prec >= SEL["min_precision"] and not any(x[2] == "word" and x[4] == sorted(int(t) for t in T) for x in out):
                    out.append((m, prec, "word", "word", sorted(int(x) for x in T)))
                break
    covc = (cm & rem[:, None]).mean(0)
    for c in np.argsort(-covc, kind="stable")[:SEL["candidates"]]:
        if covc[c] < PART["min_cov"] or nm[c].startswith("domain:"):
            continue
        on = cm[:, c] & rem
        vals, cs = np.unique(tk[on], return_counts=True)
        o = np.argsort(-cs, kind="stable")
        k = int(np.searchsorted(np.cumsum(cs[o]) / on.sum(), SEL["token_share"]) + 1)
        if k > SEL["max_tokens"]:
            continue
        T = vals[o[:k]]
        m = cm[:, c] & np.isin(tk, T)
        if (m & rem).mean() < PART["min_cov"]:
            continue
        tcov = np.isin(tk, T).mean()
        if m.mean() / tcov < SEL["min_lift_in_tokens"] * hist[c][T].sum() / max(int(alltok[T].sum()), 1):
            continue  # c does not matter within T: that is the word part
        prec = m.mean() * count / max(int(hist[c][T].sum()), 1)
        if prec >= SEL["min_precision"]:
            out.append((m, prec, "conj", nm[c], sorted(int(x) for x in T)))
    return out


def _key(part):
    return part["condition"] + "|" + ",".join(map(str, part["tokens"]))


def select(a):
    P, man = R3._setup(a)
    if man and man["events"]["sha256"] != R3._sha(P["events"]):
        sys.exit("the event sample differs from the one the protocol registered")
    from bonsai_interp import explore
    t0 = time.time()
    z = EV.load(P["events"])
    ids, mp = z["ids"], int(z["min_pos"])
    toks = explore.corpus()
    nm = CD.names()
    hist, _, alltok = CD.population(mp)
    count = z["count_dom"].sum(-1)
    U, inv = np.unique(ids[ids >= 0], return_inverse=True)
    print(f"{len(U):,} distinct sampled positions; evaluating {len(nm)} conditions", flush=True)
    CU = CD.at(U)
    tokU = toks[U // 1024, U % 1024].astype(np.int64)
    uidx = np.full(ids.shape, -1, np.int64)
    uidx[ids >= 0] = inv
    skip = _excluded()
    hyps = []
    for L in range(EV.LAYERS):
        elig = ct.eligibility(L)
        for p in (0, 1):
            ui = uidx[L, p]
            n_ev = (ui >= 0).sum(1)
            for n in np.nonzero((n_ev >= SEL["min_events"]) & (elig[p] == "eligible"))[0]:
                if (L, int(n), "+" if p == 0 else "-") in skip:
                    continue
                u = ui[n][ui[n] >= 0]
                if len(np.unique(U[u] // 1024)) < SEL["min_windows"]:
                    continue
                cm, tk, cnt = CU[u], tokU[u], int(count[L, p, n])
                none = np.zeros(len(u), bool)
                first = _parts(cm, tk, ~none, cnt, hist, alltok, nm)
                if not first:
                    continue
                best = None
                single = [x for x in first if x[0].mean() >= SEL["union_cov"]]
                if single:  # one part covers half the pole: that part alone is the hypothesis
                    m1, pr1, k1, c1, T1 = max(single, key=lambda x: x[0].mean() * x[1])
                    best = (m1.mean(), pr1, [(m1, pr1, k1, c1, T1)])
                for m1, pr1, k1, c1, T1 in ([] if single else sorted(first, key=lambda x: -x[0].mean())[:6]):
                    for m2, pr2, k2, c2, T2 in _parts(cm, tk, ~m1, cnt, hist, alltok, nm)[:6]:
                        if k1 == k2 == "word" or set(T1) & set(T2):
                            continue  # distinct triggers only: no shared token (two word parts are one word set)
                        uc = (m1 | m2).mean()
                        if uc >= SEL["union_cov"] and (best is None or uc * min(pr1, pr2) > best[0] * best[1]):
                            best = (uc, min(pr1, pr2), [(m1, pr1, k1, c1, T1), (m2, pr2, k2, c2, T2)])
                if best is None:
                    continue
                uc, pmin, parts = best
                hyps.append({"neuron": ct.pid(p, int(n), L), "union_coverage": round(float(uc), 3), "events": int(len(u)),
                             "score": round(float(uc * pmin), 4),
                             "parts": [{"kind": k, "condition": c, "tokens": T, "coverage": round(float(m.mean()), 3),
                                        "precision": round(float(pr), 3)} for m, pr, k, c, T in parts]})
        if L % 6 == 5:
            print(f"  through layer {L}: {len(hyps)} poles with a cover ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["neuron"]))
    cap = {k: (a.max * v // sum(SEL["cap"].values()) if a.max else v) for k, v in SEL["cap"].items()}
    kept = [h for k in (1, 2) for h in [x for x in hyps if len(x["parts"]) == k][:cap[k]]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} poles, kept {len(kept)} "
          f"({sum(len(h['parts']) == 2 for h in kept)} two-part, {sum(len(h['parts']) == 1 for h in kept)} one-part)", flush=True)


def _word_pools(split):
    """Per source of a split: the positions a snippet may end at, as flat arrays (token, frequency band x domain key,
    window, position) sorted by token, and per key a seeded pool of quiet candidates (at most 4 x CAP["class"])."""
    from bonsai_interp.grammar import posmorph as PM
    band = PM._band()
    tok, key, win, pos = [], [], [], []
    for kind, w, ids, M, Mq, can in R3._test_masks(split):
        rw, rp = np.nonzero(can)
        t = ids[rw, rp]
        tok.append(t)
        key.append(band[t] * 2 + (kind == "code"))
        win.append(w[rw])
        pos.append(rp)
    tok, key, win, pos = (np.concatenate(x) for x in (tok, key, win, pos))
    o = np.argsort(tok, kind="stable")
    tok, key, win, pos = tok[o], key[o], win[o], pos[o]
    rng = np.random.default_rng([29, 1 if split == "test" else 0])
    pools = {}
    for k in np.unique(key):
        idx = np.nonzero(key == k)[0]
        pools[int(k)] = np.sort(rng.permutation(idx)[:4 * R3.CAP["class"]])
    return tok, key, win, pos, pools


def _word_battery(items, split):
    """{key: pairs, "placebo:key": pairs} for word parts: a token of T against other tokens of its band and domain."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    tok, key, win, pos, pools = _word_pools(split)
    out = {}
    for name, T in sorted(items.items()):
        T = np.asarray(T)
        fi = np.concatenate([np.arange(*np.searchsorted(tok, [t, t + 1])) for t in T])
        qi = np.concatenate([pools[int(k)] for k in np.unique(key[fi])]) if len(fi) else np.zeros(0, int)
        qi = qi[~np.isin(tok[qi], T)]
        seed = [23, sum(map(ord, name)) % 100003, 1 if split == "test" else 0]
        rng = np.random.default_rng(seed + [5])
        F = C._cap(rng, key[fi].tolist(), win[fi], pos[fi], R3.CAP["class"])
        Q = C._cap(rng, key[qi].tolist(), win[qi], pos[qi], R3.CAP["class"], ["other token"] * len(qi))
        pairs = SK._pairs(F, Q, C.N_TEST, R3.PER_KEY["class"]["test"], seed)
        if len(pairs) < C.N_TEST:
            pairs = SK._pairs(F, Q, C.N_TEST, 4 * R3.PER_KEY["class"]["test"], seed)
        plac = C._placebo(Q, C.N_TEST, seed + [2], 4 * R3.PER_KEY["class"]["test"])

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - C.SNIP + 1:e.p + 1])
        out[name] = [(snip(f), snip(q), st) for f, q, st in pairs]
        out["placebo:" + name] = [(snip(x), snip(y), st) for x, y, st in plac]
    return out


def build(a):
    P, man = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    conj = {_key(pt): (pt["condition"], pt["tokens"]) for h in sel for pt in h["parts"] if pt["kind"] == "conj"}
    word = {_key(pt): pt["tokens"] for h in sel for pt in h["parts"] if pt["kind"] == "word"}
    t0 = time.time()
    bat = {**(R3.battery(conj, split) if conj else {}), **(_word_battery(word, split) if word else {})}
    print(f"batteries for {len(conj)} conjunction and {len(word)} word item sets in {time.time() - t0:.0f} s", flush=True)
    tk = explore.tokenizer()
    rows, tested = [], set()
    for key in sorted(set(conj) | set(word)):
        T = conj[key][1] if key in conj else word[key]
        words = ", ".join(repr(tk.decode([t])) for t in T)
        lab = (f"compose (one part of a cover): fires on {words} where {conj[key][0]} holds, not on the same tokens elsewhere"
               if key in conj else f"compose (one part of a cover): fires on {words}, not on other tokens of the same frequency")
        pairs, plac = bat[key], bat["placebo:" + key]
        item = {"type": "items", "kind": KIND, "condition": key, "label": lab}
        if len(pairs) < C.N_TEST:
            rows.append({**item, "class": "untestable", "why": f"only {len(pairs)} held-out pairs"})
            continue
        tested.add(key)
        rows.append({**item, "class": "tested", "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                     "quiet_kind": [st for _, _, st in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": KIND, "condition": "placebo:" + key, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
    n_poles = 0
    for h in sel:
        keys = [_key(pt) for pt in h["parts"]]
        if not all(k in tested for k in keys):
            continue  # a cover with an untestable part cannot be assigned: none of its parts are tested
        n_poles += 1
        for pt, k in zip(h["parts"], keys):
            rows.append({"type": "test", "neuron": h["neuron"], "kind": KIND, "condition": k,
                         "select": {"part_kind": pt["kind"], "coverage": pt["coverage"], "precision": pt["precision"],
                                    "union_coverage": h["union_coverage"], "parts": len(keys)}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "poles": n_poles, "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def report(a):
    P = R3._paths(a.smoke) if a.smoke else R3._paths()
    from bonsai_interp.graph.protocols import v2_stats
    res = ct._read_gz(P["results"])
    sel = {h["neuron"]: h for h in ct._read_gz(P["select"])[0]["hypotheses"]}
    pre = json.loads(P["prereg"].read_text(encoding="utf-8")) if P["prereg"].exists() else {}
    band = {L: f"{lo}-{hi}" for lo, hi in GR.BANDS for L in range(lo, hi + 1)}
    by_pole = {}
    plac = [0, 0]
    for r in res:
        if r["placebo"]:
            plac[0] += 1
            plac[1] += r["outcome"] == "held"
            continue
        by_pole.setdefault(r["neuron"], []).append(r)
    tab, rows = {}, []
    for nid, rs in by_pole.items():
        h = sel[nid]
        ok = all(r["outcome"] == "held" for r in rs)
        L = GR._neuron(nid)[0]
        for k in (("all",), ("parts", str(len(rs))), ("band", band[L])):
            e = tab.setdefault(" ".join(k), {"poles": 0, "assigned": 0, "part_tests": 0, "parts_held": 0})
            e["poles"] += 1
            e["assigned"] += ok
            e["part_tests"] += len(rs)
            e["parts_held"] += sum(r["outcome"] == "held" for r in rs)
        if ok:
            rows.append({"neuron": nid, "union_coverage": h["union_coverage"], "parts": h["parts"],
                         "tests": [{"condition": r["condition"], "label": r["label"], "auroc": round(v2_stats(r)[0], 3),
                                    "median_fire": round(v2_stats(r)[1], 3)} for r in rs]})
    ct._write_gz(P["coverage"], rows)
    summ = {"protocol": PROTOCOL, "kind": KIND, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": ct._commit(), "prereg_commit": (GR.R._alone(P["prereg"]) or "")[:12] if not a.smoke else None,
            "poles_tested": len(by_pole), "poles_assigned": len(rows), "part_tests": sum(len(v) for v in by_pole.values()),
            "parts_held": sum(r["outcome"] == "held" for v in by_pole.values() for r in v),
            "placebo_tests": plac[0], "placebo_held": plac[1], "breakdown": tab,
            "assigned_by_layer": np.bincount([GR._neuron(r["neuron"])[0] for r in rows], minlength=36).tolist() if rows else [0] * 36,
            "results_file": P["results"].name, "results_sha256": hashlib.sha256(gzip.open(P["results"], "rb").read()).hexdigest(),
            "coverage_file": P["coverage"].name, "tests_sha256": pre.get("tests_sha256")}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("poles_tested", "poles_assigned", "part_tests", "parts_held", "placebo_tests",
                                           "placebo_held", "breakdown")}, indent=1))


def protocol(a):
    R3.protocol(a)
    p = R3._paths()["protocol"]
    man = json.loads(p.read_text(encoding="utf-8"))
    man.update({"about": __doc__, "selection": SEL, "part": PART,
                "excluded": "assignment-grade poles (atlas, after compose@6) and every pole compose@3-6 tested",
                "assignment": "a pole is assignment-grade when every part of its cover held; its coverage is the union's"})
    p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["protocol", "select", "build", "score", "report"])
    ap.add_argument("--smoke")
    ap.add_argument("--events")
    ap.add_argument("--max", type=int)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    _install()
    if a.cmd == "score":
        R3._setup(a)
        return GR.score(a)
    {"protocol": protocol, "select": select, "build": build, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
