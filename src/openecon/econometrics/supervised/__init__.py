"""Resident, explicitly partitioned supervised prediction procedures."""

ESTIMATORS = ()
EXPORTS = {
    "prediction_split": "openecon.econometrics.supervised.split:prediction_split",
    "split_restore": "openecon.econometrics.supervised.split:split_restore",
    "cart": "openecon.econometrics.supervised.cart:cart",
    "cart_restore": "openecon.econometrics.supervised.cart:cart_restore",
    "cart_predict": "openecon.econometrics.supervised.cart:cart_predict",
    "cart_prediction_restore": "openecon.econometrics.supervised.cart:cart_prediction_restore",
}
