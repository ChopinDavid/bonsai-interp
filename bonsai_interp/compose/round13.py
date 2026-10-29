"""compose@13: label precision. compose@7-12 assigned word covers ("fires on these tokens") against other tokens of the
same frequency, an easy null: a pole that fires on a whole category passes it with any few of the category's commonest
members. This round tests each assigned word cover against the tokens the model itself puts nearest to it.

  python -m bonsai_interp.compose.round13 protocol | select | build | score | report   (same stages and flags as round7.py)

The engine is compose@7's (round7.py): the same test split, reader and judge (quick-8+8@2). Selection is every pole
assignment-grade (atlas, after compose@12) by a one-part word cover of compose@7, 8, 9 or 12 (its token set T). No
activation is read to choose the neighbours:
  - the neighbours of T: for each token of T, its SEL["k"] nearest tokens by cosine in the model's input embedding,
    among tokens seen >= SEL["min_count"] times in the corpus, leaving out T and every form of a token of T (the same
    letters and digits ignoring case and punctuation, or one a prefix of the other when the shorter has >= SEL["min_stem"]
    characters: inflections and glued variants, which the twin pass tested);
  - per pole, two tests on the same 8 fire snippets (a token of T):
      neighbour   against 8 snippets ending on a neighbour (same domain), with a placebo (two neighbour snippets);
      control     against 8 snippets ending on another token of the fire token's frequency band and domain (compose@7's
                  null, fresh pairs), so a neighbour failure is not resampling noise.
Verdict per pole: sharp (both held: T, not the category around it), broad (control held, neighbour failed: the pole
fires on the nearest tokens too, so T is a sample of a wider class), unreplicated (control failed: no verdict),
untestable (fewer than 8 neighbour or control pairs). Assignment does not change: a broad cover still fires on T.
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
from bonsai_interp.grammar import cons as C
from bonsai_interp.grammar import run as GR
from bonsai_interp.sweeplib import kinds as SK

SEL = {"sources": [7, 8, 9, 12], "k": 8, "min_count": 20, "min_stem": 3, "scan": 64, "cap": 20000}
QUIET = ("embedding neighbour", "other token")
EXCLUDED = "every pole not assignment-grade by a one-part word cover (atlas, after compose@12)"


def _install():
    R7.PROTOCOL, R7.BASE = "compose@13", "bonsai1.compose13"
    R7.select, R7.build, R7.report = select, build, report
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.append(Path(__file__)))


def _form(s):
    """A token's letters and digits, lower case (its stripped text when it has none)."""
    a = "".join(ch for ch in s.lower() if ch.isalnum())
    return a or s.strip()


def _same_word(a, b):
    return a == b or (min(len(a), len(b)) >= SEL["min_stem"] and (a.startswith(b) or b.startswith(a)))


def _pole(nid):
    L, p, n = GR._neuron(nid)
    return L, n, "+" if p == 0 else "-"


def _embedding():
    """The input embedding (Bonsai's token_embd, 1-bit with fp16 block scales), rows normalised."""
    import torch
    from bonsai_interp import gguf_weights as GW
    from bonsai_interp import model as MD
    for n, t in GW.tensors(MD._model_dir("BONSAI1_PATH", "Bonsai-8B")):
        if n == "model.embed_tokens.weight":
            return torch.nn.functional.normalize(GW.dense(t), dim=1)
    raise KeyError("no embedding in the model folder")


def _neighbours(covers):
    """{token: [(neighbour, cosine)]}, the SEL["scan"] nearest candidates of every token in the covers."""
    import torch
    from bonsai_interp import explore
    E = _embedding()
    cnt = np.bincount(explore.corpus().ravel().astype(np.int64), minlength=E.shape[0])[:E.shape[0]]
    Ec = E.clone()
    Ec[torch.from_numpy(cnt < SEL["min_count"])] = 0
    U = sorted({t for T in covers for t in T})
    out = {}
    for i in range(0, len(U), 2048):
        q = U[i:i + 2048]
        top = (E[torch.tensor(q)] @ Ec.T).topk(SEL["scan"], dim=1)
        for j, t in enumerate(q):
            out[t] = list(zip(top.indices[j].tolist(), [round(v, 4) for v in top.values[j].tolist()]))
    return out


def select(a):
    P, _ = R3._setup(a)
    from bonsai_interp import explore
    t0 = time.time()
    asg = R3._assigned()
    hyps = []
    for r in SEL["sources"]:
        for row in ct._read_gz(R3.OUT / f"bonsai1.compose{r}.coverage.jsonl.gz"):
            pt = row["parts"]
            if len(pt) == 1 and pt[0]["kind"] == "word" and _pole(row["neuron"]) in asg:
                hyps.append({"neuron": row["neuron"], "source": f"compose@{r}", "tokens": pt[0]["tokens"],
                             "coverage": pt[0]["coverage"], "precision": pt[0]["precision"]})
    tk = explore.tokenizer()
    nb = _neighbours([h["tokens"] for h in hyps])
    form = {}

    def f(t):
        if t not in form:
            form[t] = _form(tk.decode([t]))
        return form[t]
    for h in hyps:
        T = set(h["tokens"])
        fs = [f(t) for t in T]
        N = {}
        for t in h["tokens"]:
            got = 0
            for c, v in nb[t]:
                if got == SEL["k"]:
                    break
                if c in T or any(_same_word(f(c), x) for x in fs):
                    continue
                N[c] = max(v, N.get(c, -1.0))
                got += 1
        h["neighbours"] = sorted(N)
        h["neighbour_cos"] = round(float(np.mean(list(N.values()))), 4) if N else None
    hyps.sort(key=lambda h: GR._neuron(h["neuron"]))
    kept = hyps[::max(1, len(hyps) // a.max)][:a.max] if a.max else hyps[:SEL["cap"]]
    ct._write_gz(P["select"], [{"hypotheses": kept, "n_all": len(hyps), "minutes": round((time.time() - t0) / 60, 1)}])
    print(f"wrote {P['select']}: {len(hyps)} assigned word covers, kept {len(kept)}", flush=True)


def _batteries(sel, split):
    """{key: pairs, "placebo:key": pairs} for every cover: a neighbour test and a control on the same fire snippets."""
    from bonsai_interp import explore
    toks, dom = explore.corpus(), ct._domains()
    tok, key, win, pos, pools = R7._word_pools(split)
    dk = key % 2  # domain: English 0, code 1

    def at(T):
        return np.concatenate([np.arange(*np.searchsorted(tok, [t, t + 1])) for t in T]) if len(T) else np.zeros(0, int)

    def snip(w, p):
        return SK.Snip(w, p, int(dom[w]), toks[w, p - C.SNIP + 1:p + 1])
    out = {}
    for T, N in sorted({(tuple(h["tokens"]), tuple(h["neighbours"])) for h in sel}):
        name = ",".join(map(str, T))
        seed = [31, sum(map(ord, name)) % 100003, 1 if split == "test" else 0]
        rng = np.random.default_rng(seed + [5])
        fi, qi = at(T), at(N)
        F = C._cap(rng, dk[fi].tolist(), win[fi], pos[fi], R3.CAP["class"])
        Q = C._cap(rng, dk[qi].tolist(), win[qi], pos[qi], R3.CAP["class"], [QUIET[0]] * len(qi))
        pairs = SK._pairs(F, Q, C.N_TEST, C.N_TEST, seed)
        plac = C._placebo(Q, C.N_TEST, seed + [2], C.N_TEST)
        band = dict(zip(zip(win[fi].tolist(), pos[fi].tolist()), key[fi].tolist()))
        used = {w for f, q, _ in pairs for w in (f.w, q.w)}
        ctl, crng = [], np.random.default_rng(seed + [7])
        for f, _, _ in pairs:
            pool = pools[band[(f.w, f.p)]]
            for i in pool[crng.permutation(len(pool))]:
                if win[i] not in used and tok[i] not in T:
                    used.add(int(win[i]))
                    ctl.append((snip(f.w, f.p), snip(int(win[i]), int(pos[i])), QUIET[1]))
                    break
        out["neighbour|" + name] = [(snip(f.w, f.p), snip(q.w, q.p), s) for f, q, s in pairs]
        out["placebo:neighbour|" + name] = [(snip(x.w, x.p), snip(y.w, y.p), s) for x, y, s in plac]
        out["control|" + name] = ctl
    return out


def build(a):
    P, _ = R3._setup(a)
    from bonsai_interp import explore
    from bonsai_interp.sweeplib import run as R
    split = "select" if a.smoke else "test"
    sel = ct._read_gz(P["select"])[0]["hypotheses"]
    t0 = time.time()
    bat = _batteries(sel, split)
    print(f"batteries for {len(bat) // 3} token sets in {time.time() - t0:.0f} s", flush=True)
    tk = explore.tokenizer()

    def words(T, n=99):
        return ", ".join(repr(tk.decode([t])) for t in T[:n]) + (", ..." if len(T) > n else "")
    rows, tested = [], set()
    for T, N in sorted({(tuple(h["tokens"]), tuple(h["neighbours"])) for h in sel}):
        name = ",".join(map(str, T))
        kn, kc = "neighbour|" + name, "control|" + name
        labs = {kn: f"compose (precision): fires on {words(T)}, not on their nearest tokens in the model's embedding ({words(N, 12)})",
                kc: f"compose (replication): fires on {words(T)}, not on other tokens of the same frequency (fresh pairs)"}
        if len(bat[kn]) < C.N_TEST or len(bat[kc]) < C.N_TEST:
            for k in (kn, kc):
                rows.append({"type": "items", "kind": R7.KIND, "condition": k, "label": labs[k], "class": "untestable",
                             "why": f"only {len(bat[kn])} neighbour and {len(bat[kc])} control pairs"})
            continue
        tested.add(name)
        for k in (kn, kc):
            pairs = bat[k]
            rows.append({"type": "items", "kind": R7.KIND, "condition": k, "label": labs[k], "class": "tested",
                         "fire": [R._text(f) for f, _, _ in pairs], "quiet": [R._text(q) for _, q, _ in pairs],
                         "quiet_kind": [s for _, _, s in pairs], "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in pairs]})
        plac = bat["placebo:" + kn]
        if len(plac) >= C.N_TEST:
            rows.append({"type": "items", "kind": R7.KIND, "condition": "placebo:" + kn, "label": "placebo", "class": "placebo",
                         "fire": [R._text(f) for f, _, _ in plac], "quiet": [R._text(q) for _, q, _ in plac],
                         "quiet_kind": ["placebo"] * len(plac), "windows": [[f.w, f.p, q.w, q.p] for f, q, _ in plac]})
    n_poles = 0
    for h in sel:
        name = ",".join(map(str, h["tokens"]))
        if name not in tested:
            continue
        n_poles += 1
        s = {"source": h["source"], "items": len(h["tokens"]), "neighbours": len(h["neighbours"]),
             "neighbour_cos": h["neighbour_cos"], "coverage": h["coverage"], "precision": h["precision"]}
        for k in ("neighbour|", "control|"):
            rows.append({"type": "test", "neuron": h["neuron"], "kind": R7.KIND, "condition": k + name, "select": s})
    sha = ct._write_gz(P["tests"], rows)
    counts = {"items": len(tested), "untestable": sum(r["type"] == "items" and r["class"] == "untestable" for r in rows) // 2,
              "poles": n_poles, "tests": sum(r["type"] == "test" for r in rows)}
    pre = {"protocol": R7.PROTOCOL, "protocol_commit": GR.registered() if not a.smoke else None,
           "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tests_file": P["tests"].name, "tests_sha256": sha,
           "select_file": P["select"].name, "select_sha256": hashlib.sha256(gzip.open(P["select"], "rb").read()).hexdigest(),
           "counts": counts}
    P["prereg"].write_text(json.dumps(pre, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(counts), f"\nwrote {P['tests']} and {P['prereg']}: commit the manifest ALONE before scoring", flush=True)


def _bin(x, edges, names):
    return names[int(np.searchsorted(edges, x, side="right"))]


def report(a):
    P = R3._paths(a.smoke) if a.smoke else R3._paths()
    from bonsai_interp.graph.protocols import v2_stats
    res = ct._read_gz(P["results"])
    pre = json.loads(P["prereg"].read_text(encoding="utf-8")) if P["prereg"].exists() else {}
    band = {L: f"{lo}-{hi}" for lo, hi in GR.BANDS for L in range(lo, hi + 1)}
    by_pole, plac = {}, [0, 0]
    for r in res:
        if r["placebo"]:
            plac[0] += 1
            plac[1] += r["outcome"] == "held"
            continue
        by_pole.setdefault(r["neuron"], {})[r["condition"].split("|")[0]] = r
    tab, rows = {}, []
    for nid, d in sorted(by_pole.items()):
        n, c = d["neighbour"], d["control"]
        v = ("sharp" if n["outcome"] == "held" else "broad") if c["outcome"] == "held" else "unreplicated"
        s = n["select"]
        for k in ("all", "band " + band[GR._neuron(nid)[0]], "source " + s["source"],
                  "items " + _bin(s["items"], [1, 3, 8], ["1", "2-3", "4-8", "9-16"]),
                  "neighbour_cos " + _bin(s["neighbour_cos"] or 0, [0.1, 0.13, 0.16], ["<0.10", "0.10-0.13", "0.13-0.16", ">=0.16"])):
            e = tab.setdefault(k, {"poles": 0, "sharp": 0, "broad": 0, "unreplicated": 0})
            e["poles"] += 1
            e[v] += 1
        st = {k: v2_stats(r) for k, r in d.items()}
        rows.append({"neuron": nid, "verdict": v, "select": s, "label": n["label"],
                     "tests": {k: {"outcome": d[k]["outcome"], "auroc": round(st[k][0], 3), "median_fire": round(st[k][1], 3),
                                   "median_quiet": round(float(np.median([x["act"] for x in d[k]["per_sentence"] if x["set"] == "quiet"])), 3)}
                               for k in d}})
    ct._write_gz(P["coverage"], rows)
    tot = tab.get("all", {})
    summ = {"protocol": R7.PROTOCOL, "kind": R7.KIND, "scored": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "toolchain_commit": ct._commit(), "prereg_commit": (GR.R._alone(P["prereg"]) or "")[:12] if not a.smoke else None,
            "poles_tested": len(by_pole), "sharp": tot.get("sharp", 0), "broad": tot.get("broad", 0),
            "unreplicated": tot.get("unreplicated", 0), "placebo_tests": plac[0], "placebo_held": plac[1], "breakdown": tab,
            "sharp_by_layer": np.bincount([GR._neuron(r["neuron"])[0] for r in rows if r["verdict"] == "sharp"], minlength=36).tolist(),
            "broad_by_layer": np.bincount([GR._neuron(r["neuron"])[0] for r in rows if r["verdict"] == "broad"], minlength=36).tolist(),
            "results_file": P["results"].name, "results_sha256": hashlib.sha256(gzip.open(P["results"], "rb").read()).hexdigest(),
            "coverage_file": P["coverage"].name, "tests_sha256": pre.get("tests_sha256")}
    P["summary"].write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summ[k] for k in ("poles_tested", "sharp", "broad", "unreplicated", "placebo_tests", "placebo_held",
                                           "breakdown")}, indent=1))


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about": __doc__, "selection": SEL, "excluded": EXCLUDED, "quiet": list(QUIET),
                    "verdict": "sharp: both tests held; broad: control held, neighbour failed; unreplicated: control failed",
                    "assignment": "unchanged: the verdict qualifies an assigned word cover, it does not assign or unassign"})
        man.pop("part", None)
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
