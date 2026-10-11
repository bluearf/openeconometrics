"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720011


def make_data():
    return pd.DataFrame(
        {
            "output": [20.0, 40.0, 60.0, 80.0, 100.0, 60.0],
            "wage": [4.0, 4.0, 4.0, 4.0, 4.0, 9.0],
            "rental": [1.0] * 6,
        }
    )
