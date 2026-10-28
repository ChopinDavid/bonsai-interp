"""compose@9: compose@8 (round8.py, unchanged) on the rest of the queue: every remaining one-part cover.

  python -m bonsai_interp.compose.round9 protocol | select | build | score | report   (same stages and flags as round7.py)

compose@8 assigned 6,353 of 7,215 one-part covers ranked 1-8,000 after compose@7's (88%, against compose@7's 98%). This
round tests every one-part cover left in the queue (cap 40,000), so the hold rate's fall down the ranking is measured to
the end. Poles compose@3-8 tested are left out, as well as every assignment-grade pole (atlas, after compose@8).
"""
import json
from pathlib import Path

from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round5 as R5
from bonsai_interp.compose import round7 as R7
from bonsai_interp.compose import round8 as R8

EXCLUDED = "assignment-grade poles (atlas, after compose@8) and every pole compose@3-8 tested"


def _install():
    R8._install()
    R7.PROTOCOL, R7.BASE = "compose@9", "bonsai1.compose9"
    R7.SEL = {**R7.SEL, "cap": {1: 40000, 2: 0}}
    R7._excluded = lambda: R3._assigned() | R5._tested(3, 4, 5, 6, 7, 8)
    install8 = R7._install
    R7._install = lambda: (install8(), R3.CODE.append(Path(__file__)))


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about9": __doc__, "selection": R7.SEL, "excluded": EXCLUDED})
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
