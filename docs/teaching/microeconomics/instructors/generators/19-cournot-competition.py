"""Original synthetic observations, for instructor preparation only."""

import pandas as pd

SEED = 720019


def make_data():
    return pd.DataFrame({"firms": range(1, 11), "intercept": 100.0, "slope": 1.0, "mc": 20.0})
