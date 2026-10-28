"""compose@10: pair covers. compose@9 found that a cover's selection precision predicts its hold (11% at 0.1-0.2, 70% at
0.5 or more), and the occlusion pilot (FINDINGS 2026-10-26) that a peak is mostly the read token and the one before it.
A pair cover names both: the pole fires on these (previous token, token) pairs.

  python -m bonsai_interp.compose.round10 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py): the same event sample (pinned by hash), test split, reader, judge and report.
Selection, per eligible pole not assignment-grade (atlas, after compose@9), with >= 32 sampled activations in >= 8
windows. Poles earlier rounds tested are eligible: a pair cover is a new hypothesis, and each pole gets one.
  - the part is the fewest of the pole's commonest (previous token, token) pairs (at most 8) that cover half of its
    sampled strong activations;
  - precision (the pairs' share of the pole's strong count over their corpus occurrences, as in compose@3) >= 0.3;
  - the previous token must matter: the pair precision is >= 1.5x the precision of the same read tokens alone;
  - hypotheses are ranked by coverage x precision; the best SEL["cap"] are tested.
Test (quick-8+8@2, placebo as in compose@3): 8 snippets ending on one of the pairs, against 8 near misses from two
strata, the same read token after another previous token, and the same previous token before another token. Snippets
end where a word ends (compose@3's rule), split by domain (English, code).
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

SEL = {"min_events": 32, "min_windows": 8, "cov": 0.5, "min_precision": 0.3, "pair_over_token": 1.5, "max_pairs": 8,
       "cap": 12000}
EXCLUDED = "assignment-grade poles (atlas, after compose@9); poles earlier rounds tested are eligible"
QUIET = ("other previous", "other token")


def _install():
    R7.PROTOCOL, R7.BASE = "compose@10", "bonsai1.compose10"
    R7.select, R7.build = select, build
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.append(Path(__file__)))


def _V():
    return ct.vocab()["V"]


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
    V = _V()
    T = toks[z["windows"].astype(np.int64)].astype(np.int64)
    pc = (T[:, :-1] * V + T[:, 1:])[:, mp - 1:].ravel()  # every pair ending at a counted position
    upair, ucount = np.unique(pc, return_counts=True)
    tokc = np.bincount(T[:, mp:].ravel(), minlength=V)
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
                tk = toks[w, q].astype(np.int64)
                k = toks[w, q - 1].astype(np.int64) * V + tk
                vals, cs = np.unique(k, return_counts=True)
                o = np.argsort(-cs, kind="stable")
                cum = np.cumsum(cs[o]) / len(k)
                m = int(np.searchsorted(cum, SEL["cov"]) + 1)
                if m > SEL["max_pairs"]:
                    continue
                Pk = np.sort(vals[o[:m]])
                cov = float(np.isin(k, Pk).mean())
                prec = cov * count[L, p, n] / max(int(ucount[np.searchsorted(upair, Pk)].sum()), 1)
                Tk = np.unique(Pk % V)
                tprec = float(np.isin(tk, Tk).mean()) * count[L, p, n] / max(int(tokc[Tk].sum()), 1)
                if prec < SEL["min_precision"] or prec < SEL["pair_over_token"] * tprec:
                    continue
                hyps.append({"neuron": ct.pid(p, int(n), L), "union_coverage": round(cov, 3), "events": int(len(e)),
                             "score": round(float(cov * prec), 4), "token_precision": round(float(tprec), 3),
                             "parts": [{"kind": "pair", "condition": "pair", "tokens": [int(x) for x in Pk],
                                        "coverage": round(cov, 3), "precision": round(float(prec), 3)}]})
        if L % 6 == 5:
            print(f"  through layer {L}: {len(hyps)} poles with a pair cover ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["neuron"]))
    kept = hyps[:a.max or SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} poles, kept {len(kept)}", flush=True)


def _pair_battery(items, split):
    """{key: pairs, "placebo:key": pairs}: a snippet ending on one of the pairs, against the same read token after another
    previous token or the same previous token before another token."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    V = _V()
    tok, key, win, pos, _ = R7._word_pools(split)  # every position a snippet may end at, sorted by token
    prev = toks[win, pos - 1].astype(np.int64)
    pk = prev * V + tok
    dk = key % 2  # domain: English 0, code 1
    op = np.argsort(prev, kind="stable")
    prev_s = prev[op]
    out = {}
    for name, Pk in sorted(items.items()):
        Pk = np.asarray(Pk, np.int64)
        Tk, Pv = np.unique(Pk % V), np.unique(Pk // V)
        seed = [23, sum(map(ord, name)) % 100003, 1 if split == "test" else 0]
        rng = np.random.default_rng(seed + [5])
        it = np.concatenate([np.arange(*np.searchsorted(tok, [t, t + 1])) for t in Tk])
        on = np.isin(pk[it], Pk)
        fi, q1 = it[on], it[~on]
        ip = np.concatenate([op[slice(*np.searchsorted(prev_s, [v, v + 1]))] for v in Pv])
        if len(ip) > 8 * R3.CAP["class"]:
            ip = rng.choice(ip, 8 * R3.CAP["class"], replace=False)
        q2 = ip[~np.isin(tok[ip], Tk)]
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
    bat = _pair_battery(items, split)
    print(f"batteries for {len(items)} pair sets in {time.time() - t0:.0f} s", flush=True)
    tk, V = explore.tokenizer(), _V()
    rows, tested = [], set()
    for key in sorted(items):
        pairs_txt = ", ".join(repr(tk.decode([x // V]) + "|" + tk.decode([x % V])) for x in items[key])
        lab = (f"compose (pair cover): fires on the (previous token | token) pairs {pairs_txt}, not on the same token after "
               f"another token, nor the same previous token before another token")
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
                     "select": {"part_kind": "pair", "coverage": pt["coverage"], "precision": pt["precision"],
                                "token_precision": h["token_precision"], "union_coverage": h["union_coverage"], "parts": 1}})
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
                    "assignment": "a pole is assignment-grade when its pair cover held; its coverage is the cover's"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
