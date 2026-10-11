"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720012


def make_data():
    return pd.DataFrame(
        {
            "price": [float(i) for i in range(1, 21)],
            "fixed_cost": 100.0,
            "linear_cost": 2.0,
            "quadratic_cost": 0.1,
        }
    )
