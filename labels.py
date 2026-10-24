"""Graph-level labels derived from the dataset's folder / file naming."""

import re
from pathlib import Path


def graph_label_from_path(path):
    """Return 0 (clean), 1 (Trojan-infected) or -1 (unknown).

    Rules:
      * anything under a folder named TjFree           -> 0
      * file stem ending in _T000 (golden circuit)     -> 0
      * file stem ending in _T### with ### != 000      -> 1
    """
    p = Path(path)
    if "tjfree" in [part.lower() for part in p.parts]:
        return 0
    m = re.search(r"_T(\d+)$", p.stem, re.IGNORECASE)
    if m:
        return 0 if int(m.group(1)) == 0 else 1
    return -1
