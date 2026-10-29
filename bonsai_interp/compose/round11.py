"""compose@11: triple covers. compose@10's pair covers (round10.py) with one more token of context: the pole fires on
these (token two back, previous token, token) triples. The occlusion pilot (FINDINGS 2026-10-26) found the token two
back carrying 18-40% of a peak, most in layers 18-23.

  python -m bonsai_interp.compose.round11 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py), as in compose@10. Selection, per eligible pole not assignment-grade (atlas, after
compose@10), with >= 32 sampled activations in >= 8 windows (poles earlier rounds tested are eligible):
  - the part is the fewest of the pole's commonest triples (at most 8) that cover half of its sampled strong activations;
  - precision >= 0.3, and >= 1.5x the precision of the same (previous token, token) pairs alone, so the token two back
    matters;
  - hypotheses are ranked by coverage x precision; the best SEL["cap"] are tested.
Test (quick-8+8@2, placebo as in compose@3): 8 snippets ending on one of the triples, against 8 near misses from two
strata, the same pair after another token two back, and the same two tokens before another token.
"""
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from bonsai_interp.auto import context_tier as ct
from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round7 as R7
from bonsai_interp.events import sample as EV
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR
from bonsai_interp.sweeplib import kinds as SK

SEL = {"min_events": 32, "min_windows": 8, "cov": 0.5, "min_precision": 0.3, "triple_over_pair": 1.5, "max_triples": 8,
       "cap": 12000}
EXCLUDED = "assignment-grade poles (atlas, after compose@10); poles earlier rounds tested are eligible"
QUIET = ("other token two back", "other token")


def _install():
    R7.PROTOCOL, R7.BASE = "compose@11", "bonsai1.compose11"
    R7.select, R7.build = select, build
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.append(Path(__file__)))


def select(a):
    P, man = R3._setup(a)
    if man and man["events"]["sha256"] != R3._sha(P["events"]):
        sys.exit("the event sample differs from the one the protocol registered")
    from bonsai_interp import explore
    t0 = time.time()
    z = EV.load(P["events"])
    ids, mp = z["ids"], int(z["min_pos"])
    count = z["count_dom"].sum(-1)
    toks = explore.corpus()
    V = ct.vocab()["V"]
    T = toks[z["windows"].astype(np.int64)].astype(np.int64)
    ut, uc = np.unique(((T[:, :-2] * V + T[:, 1:-1]) * V + T[:, 2:])[:, mp - 2:].ravel(), return_counts=True)
    up, upc = np.unique((T[:, :-1] * V + T[:, 1:])[:, mp - 1:].ravel(), return_counts=True)
    skip = R3._assigned()
    hyps = []
    for L in range(EV.LAYERS):
        elig = ct.eligibility(L)
        for p in (0, 1):
            ev = ids[L, p]
            for n in np.nonzero(((ev >= 0).sum(1) >= SEL["min_events"]) & (elig[p] == "eligible"))[0]:
                if (L, int(n), "+" if p == 0 else "-") in skip:
                    continue
                e = ev[n][ev[n] >= 0].astype(np.int64)
                w, q = e // 1024, e % 1024
                if len(np.unique(w)) < SEL["min_windows"]:
                    continue
                pk = toks[w, q - 1].astype(np.int64) * V + toks[w, q].astype(np.int64)
                k = toks[w, q - 2].astype(np.int64) * V * V + pk
                vals, cs = np.unique(k, return_counts=True)
                o = np.argsort(-cs, kind="stable")
                m = int(np.searchsorted(np.cumsum(cs[o]) / len(k), SEL["cov"]) + 1)
                if m > SEL["max_triples"]:
                    continue
                Pk = np.sort(vals[o[:m]])
                cov = float(np.isin(k, Pk).mean())
                prec = cov * count[L, p, n] / max(int(uc[np.searchsorted(ut, Pk)].sum()), 1)
                Pp = np.unique(Pk % (V * V))
                pprec = float(np.isin(pk, Pp).mean()) * count[L, p, n] / max(int(upc[np.searchsorted(up, Pp)].sum()), 1)
                if prec < SEL["min_precision"] or prec < SEL["triple_over_pair"] * pprec:
                    continue
                hyps.append({"neuron": ct.pid(p, int(n), L), "union_coverage": round(cov, 3), "events": int(len(e)),
                             "score": round(float(cov * prec), 4), "pair_precision": round(float(pprec), 3),
                             "parts": [{"kind": "triple", "condition": "triple", "tokens": [int(x) for x in Pk],
                                        "coverage": round(cov, 3), "precision": round(float(prec), 3)}]})
        if L % 6 == 5:
            print(f"  through layer {L}: {len(hyps)} poles with a triple cover ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["neuron"]))
    kept = hyps[:a.max or SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} poles, kept {len(kept)}", flush=True)


def _triple_battery(items, split):
    """{key: pairs, "placebo:key": pairs}: a snippet ending on one of the triples, against the same (previous token,
    token) pair after another token two back or the same two tokens before another token."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    V = ct.vocab()["V"]
    tok, key, win, pos, _ = R7._word_pools(split)  # every position a snippet may end at
    prev, prev2 = toks[win, pos - 1].astype(np.int64), toks[win, pos - 2].astype(np.int64)
    pk = prev * V + tok          # the pair ending here
    hk = prev2 * V + prev        # the two tokens before here
    tk3 = prev2 * V * V + pk
    dk = key % 2  # domain: English 0, code 1
    o1, o2 = np.argsort(pk, kind="stable"), np.argsort(hk, kind="stable")
    s1, s2 = pk[o1], hk[o2]
    out = {}
    for name, Pk in sorted(items.items()):
        Pk = np.asarray(Pk, np.int64)
        Pp, Ph = np.unique(Pk % (V * V)), np.unique(Pk // V)
        seed = [23, sum(map(ord, name)) % 100003, 1 if split == "test" else 0]
        rng = np.random.default_rng(seed + [5])
        i1 = np.concatenate([o1[slice(*np.searchsorted(s1, [v, v + 1]))] for v in Pp])
        on = np.isin(tk3[i1], Pk)
        fi, q1 = i1[on], i1[~on]
        i2 = np.concatenate([o2[slice(*np.searchsorted(s2, [v, v + 1]))] for v in Ph])
        if len(i2) > 8 * R3.CAP["class"]:
            i2 = rng.choice(i2, 8 * R3.CAP["class"], replace=False)
        q2 = i2[~np.isin(pk[i2], Pp)]
        qi = np.concatenate([q1, q2])
        st = [QUIET[0]] * len(q1) + [QUIET[1]] * len(q2)
        F = C._cap(rng, dk[fi].tolist(), win[fi], pos[fi], R3.CAP["class"])
        Q = C._cap(rng, dk[qi].tolist(), win[qi], pos[qi], R3.CAP["class"], st)
        pairs = SK._pairs(F, Q, C.N_TEST, C.N_TEST, seed)
        plac = C._placebo(Q, C.N_TEST, seed + [2], C.N_TEST)

        def snip(e):
            return SK.Snip(e.w, e.p, int(dom[e.w]), toks[e.w, e.p - C.SNIP + 1:e.p + 1])
        out[name] = [(snip(f), snip(q), s) for f, q, s in pairs]
        out["placebo:" + name] = [(snip(x), snip(y), s) for x, y, s in plac]
    return out


def build(a):
    P, man = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    items = {R7._key(pt): pt["tokens"] for h in sel for pt in h["parts"]}
    t0 = time.time()
    bat = _triple_battery(items, split)
    print(f"batteries for {len(items)} triple sets in {time.time() - t0:.0f} s", flush=True)
    tk, V = explore.tokenizer(), ct.vocab()["V"]
    rows, tested = [], set()
    for key in sorted(items):
        txt = ", ".join(repr("|".join(tk.decode([t]) for t in (x // (V * V), x // V % V, x % V))) for x in items[key])
        lab = (f"compose (triple cover): fires on the (token two back | previous token | token) triples {txt}, not on the "
               f"same last two tokens after another token, nor the same first two before another token")
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
        pt = h["parts"][0]
        k = R7._key(pt)
        if k not in tested:
            continue
        n_poles += 1
        rows.append({"type": "test", "neuron": h["neuron"], "kind": R7.KIND, "condition": k,
                     "select": {"part_kind": "triple", "coverage": pt["coverage"], "precision": pt["precision"],
                                "pair_precision": h["pair_precision"], "union_coverage": h["union_coverage"], "parts": 1}})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows),
              "poles": n_poles, "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": R7.PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED, "quiet": list(QUIET),
                    "assignment": "a pole is assignment-grade when its triple cover held; its coverage is the cover's"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
