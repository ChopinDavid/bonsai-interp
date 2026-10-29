"""compose@12: wider covers. compose@7-11 limited a cover to 8 tokens, pairs or triples. This round allows 9-16 read
tokens, or 9-16 (previous token, token) pairs, for poles whose cover needs more than 8.

  python -m bonsai_interp.compose.round12 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py); tests are compose@7's word test (round7._word_battery) for token sets and
compose@10's pair test (round10._pair_battery) for pair sets. Selection, per eligible pole not assignment-grade (atlas,
after compose@11), with >= 32 sampled activations in >= 8 windows:
  - the cover is the fewest of the pole's commonest read tokens that reach half of its sampled strong activations, if
    that takes 9-16 tokens; otherwise the same with (previous token, token) pairs; a pole with a cover of at most 8
    items is left out (compose@7-10 already had that hypothesis);
  - every item is seen at least twice in the pole's sample (a set of items seen once each fits the sample, not the pole);
  - precision >= 0.3 (compose@9 and @10: holds track precision); a pair cover also needs >= 1.5x the precision of the
    same read tokens alone;
  - hypotheses are ranked by coverage x precision; the best SEL["cap"] are tested.
A held wider cover is assignment-grade by coverage, but its label is a list: a sharper label is the survey's to find.
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
from bonsai_interp.compose import round10 as R10
from bonsai_interp.events import sample as EV
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR

SEL = {"min_events": 32, "min_windows": 8, "cov": 0.5, "min_items": 9, "max_items": 16, "min_seen": 2,
       "min_precision": 0.3, "pair_over_token": 1.5, "cap": 12000}
EXCLUDED = "assignment-grade poles (atlas, after compose@11), and poles with a cover of at most 8 items"


def _install():
    R7.PROTOCOL, R7.BASE = "compose@12", "bonsai1.compose12"
    R7.select, R7.build = select, build
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.extend([Path(R10.__file__), Path(__file__)]))


def _cover(keys):
    """(sorted items, coverage, size) of the fewest commonest keys reaching SEL["cov"], or (None, 0, size)."""
    vals, cs = np.unique(keys, return_counts=True)
    o = np.argsort(-cs, kind="stable")
    m = int(np.searchsorted(np.cumsum(cs[o]) / len(keys), SEL["cov"]) + 1)
    if m > SEL["max_items"] or cs[o[m - 1]] < SEL["min_seen"]:
        return None, 0.0, m
    P = np.sort(vals[o[:m]])
    return P, float(np.isin(keys, P).mean()), m


def select(a):
    P_, man = R3._setup(a)
    if man and man["events"]["sha256"] != R3._sha(P_["events"]):
        sys.exit("the event sample differs from the one the protocol registered")
    from bonsai_interp import explore
    t0 = time.time()
    z = EV.load(P_["events"])
    ids, mp = z["ids"], int(z["min_pos"])
    count = z["count_dom"].sum(-1)
    toks = explore.corpus()
    V = ct.vocab()["V"]
    T = toks[z["windows"].astype(np.int64)].astype(np.int64)
    up, upc = np.unique((T[:, :-1] * V + T[:, 1:])[:, mp - 1:].ravel(), return_counts=True)
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
                pk = toks[w, q - 1].astype(np.int64) * V + tk
                Pt, ct_, mt = _cover(tk)
                Pp, cp, mpair = _cover(pk)
                if mt <= 8 or mpair <= 8:
                    continue  # a cover of at most 8 items: compose@7-10's hypothesis
                h = None
                if Pt is not None:
                    prec = ct_ * count[L, p, n] / max(int(tokc[Pt].sum()), 1)
                    if prec >= SEL["min_precision"]:
                        h = ("word", "word", Pt, ct_, prec)
                if h is None and Pp is not None:
                    prec = cp * count[L, p, n] / max(int(upc[np.searchsorted(up, Pp)].sum()), 1)
                    Tk = np.unique(Pp % V)
                    tprec = float(np.isin(tk, Tk).mean()) * count[L, p, n] / max(int(tokc[Tk].sum()), 1)
                    if prec >= SEL["min_precision"] and prec >= SEL["pair_over_token"] * tprec:
                        h = ("pair", "pair", Pp, cp, prec)
                if h is None:
                    continue
                kind, cond, Pk, cov, prec = h
                hyps.append({"neuron": ct.pid(p, int(n), L), "union_coverage": round(cov, 3), "events": int(len(e)),
                             "score": round(float(cov * prec), 4),
                             "parts": [{"kind": kind, "condition": cond, "tokens": [int(x) for x in Pk],
                                        "coverage": round(cov, 3), "precision": round(float(prec), 3)}]})
        if L % 6 == 5:
            print(f"  through layer {L}: {len(hyps)} poles with a wider cover ({time.time() - t0:.0f} s)", flush=True)
    hyps.sort(key=lambda h: (-h["score"], h["neuron"]))
    kept = hyps[:a.max or SEL["cap"]]
    ct._write_gz(P_["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P_['select']}: {len(hyps)} poles, kept {len(kept)} "
          f"({sum(h['parts'][0]['kind'] == 'word' for h in kept)} token sets)", flush=True)


def build(a):
    P, man = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    word = {R7._key(pt): pt["tokens"] for h in sel for pt in h["parts"] if pt["kind"] == "word"}
    pair = {R7._key(pt): pt["tokens"] for h in sel for pt in h["parts"] if pt["kind"] == "pair"}
    t0 = time.time()
    bat = {**(R7._word_battery(word, split) if word else {}), **(R10._pair_battery(pair, split) if pair else {})}
    print(f"batteries for {len(word)} token and {len(pair)} pair sets in {time.time() - t0:.0f} s", flush=True)
    tk, V = explore.tokenizer(), ct.vocab()["V"]
    rows, tested = [], set()
    for key in sorted(set(word) | set(pair)):
        if key in word:
            txt = ", ".join(repr(tk.decode([t])) for t in word[key])
            lab = f"compose (wider cover): fires on {txt}, not on other tokens of the same frequency"
        else:
            txt = ", ".join(repr(tk.decode([x // V]) + "|" + tk.decode([x % V])) for x in pair[key])
            lab = (f"compose (wider cover): fires on the (previous token | token) pairs {txt}, not on the same token after "
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
                     "select": {"part_kind": pt["kind"], "coverage": pt["coverage"], "precision": pt["precision"],
                                "items": len(pt["tokens"]), "union_coverage": h["union_coverage"], "parts": 1}})
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
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED,
                    "assignment": "a pole is assignment-grade when its wider cover held; its coverage is the cover's"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
