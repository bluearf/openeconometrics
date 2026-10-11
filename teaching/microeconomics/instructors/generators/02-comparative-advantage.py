"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720002


def make_data():
    return pd.DataFrame(
        {
            "economy": ["A", "B"],
            "labor_x": [2.0, 6.0],
            "labor_y": [4.0, 3.0],
            "hours": [120.0, 120.0],
        }
    )
