"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720008


def make_data():
    p = [2.0, 3.0, 4.0, 5.0]
    m = [100.0, 120.0, 110.0, 150.0]
    return pd.DataFrame(
        {
            "choice_id": range(1, 5),
            "px": p,
            "py": [2.0] * 4,
            "income": m,
            "x": [0.4 * a / b for a, b in zip(m, p)],
            "y": [0.6 * a / 2 for a in m],
        }
    )
