"""compose@8: compose@7's union covers (round7.py, unchanged), one-part covers only, on the next slice of the queue.

  python -m bonsai_interp.compose.round8 protocol | select | build | score | report   (same stages and flags as round7.py)

compose@7 assigned 3,629 of 3,740 one-part covers (word sets held 3,616 of 3,694) but only 34 of 1,778 two-part covers.
This round tests the next 8,000 one-part covers by the same ranking and no two-part covers. Poles compose@3-7 tested
are left out, as well as every assignment-grade pole (atlas, after compose@7). Same event sample (pinned by hash), same
library (conds4.py), same parts and tests.
"""
import json
from pathlib import Path

from bonsai_interp.compose import round3 as R3
from bonsai_interp.compose import round5 as R5
from bonsai_interp.compose import round7 as R7

EXCLUDED = "assignment-grade poles (atlas, after compose@7) and every pole compose@3-7 tested"


def _install():
    R7.PROTOCOL, R7.BASE = "compose@8", "bonsai1.compose8"
    R7.SEL = {**R7.SEL, "cap": {1: 8000, 2: 0}}
    R7._excluded = lambda: R3._assigned() | R5._tested(3, 4, 5, 6, 7)
    install7 = R7._install
    R7._install = lambda: (install7(), R3.CODE.append(Path(__file__)))


def main():
    _install()
    R7.main()
    if R7.sys.argv[1:2] == ["protocol"]:
        p = R3._paths()["protocol"]
        man = json.loads(p.read_text(encoding="utf-8"))
        man.update({"about8": __doc__, "selection": R7.SEL, "excluded": EXCLUDED})
        p.write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
