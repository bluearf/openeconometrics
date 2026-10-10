"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720006


def make_data():
    return pd.DataFrame(
        {
            "scenario": ["base", "income_low", "income_high", "x_cheaper", "x_dearer"],
            "income": [120.0, 90.0, 150.0, 120.0, 120.0],
            "px": [3.0, 3.0, 3.0, 2.0, 6.0],
            "py": [2.0] * 5,
            "alpha": [0.4] * 5,
        }
    )
