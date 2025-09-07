# pdp_ice.py
from sklearn.inspection import partial_dependence
import numpy as np

def pdp_for_features(model, X, features):
    pdp = {}
    for f in features:
        res = partial_dependence(model, X, [f], kind="average")
        pdp[f] = (res["values"][0], res["average"][0])
    return pdp
