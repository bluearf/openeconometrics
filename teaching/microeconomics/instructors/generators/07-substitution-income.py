"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720007


def make_data():
    return pd.DataFrame(
        {"px": [3.0, 6.0], "py": [2.0, 2.0], "income": [120.0, 120.0], "alpha": [0.4, 0.4]}
    )
