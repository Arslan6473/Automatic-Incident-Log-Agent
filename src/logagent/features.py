"""Early-warning feature definition, shared by training (data.py) and serving (service.py).

Kept free of pandas so the serving container stays small.
"""

import numpy as np

EARLY_K = 8
EVENT_IDS = [f"E{i}" for i in range(1, 30)]
EARLY_FEATURES = [f"early_{e}" for e in EVENT_IDS] + ["early_elapsed"]
_INDEX = {e: i for i, e in enumerate(EVENT_IDS)}


def early_vector(events: list[str], elapsed_seconds: float) -> np.ndarray:
    """Counts of each event type among the first EARLY_K events, plus elapsed seconds (order = EARLY_FEATURES)."""
    v = np.zeros(len(EARLY_FEATURES), dtype=np.float32)
    for e in events[:EARLY_K]:
        v[_INDEX[e]] += 1
    v[-1] = elapsed_seconds
    return v
