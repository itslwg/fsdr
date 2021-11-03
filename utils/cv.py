import itertools
import numpy as np

from tqdm import tqdm
from utils.data_preparation import concat_with_mi

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def generate_all_combinations(d: dict):
    """Generate all combinations of params from dict."""
    keys, values = zip(*d.items())
    return [dict(zip(keys,v)) for v in itertools.product(*values)]


def scale_and_reduce(reducer, X_tr, X_te, sparse):
    """Scale and reduce the sample."""
    ## Standardize
    standardscaler = StandardScaler()
    X_tr_scaled = standardscaler.fit_transform(X_tr)
    X_te_scaled = standardscaler.transform(X_te)
    
    # Run dimensionality reduction
    has_fit = getattr(reducer, "transform", None)
    X_tr_reduced, X_te_reduced = X_tr_scaled, X_te_scaled
    if reducer:
        if has_fit:
            X_tr_reduced = reducer.fit_transform(X_tr_scaled)
            X_te_reduced = reducer.transform(X_te_scaled)
        else:
            X_tr_scaled = X_tr_scaled.astype(np.float32)
            X_te_scaled = X_te_scaled.astype(np.float32)
            X_tr_dict = dict(X = X_tr_scaled)
            X_te_dict = dict(X = X_te_scaled)
            if sparse is not None:
                X_tr_dict['sparse'] = [sparse] * len(X_tr_scaled)
                X_te_dict['sparse'] = [sparse] * len(X_te_scaled)
            reducer = (reducer.fit(X_tr_dict, X_tr_scaled)
                       if sparse is not None
                       else reducer.fit(X_tr_scaled, X_tr_scaled))
            _, X_tr_reduced = reducer.forward(X_tr_dict) if sparse is not None else reducer.forward(X_tr_scaled)
            _, X_te_reduced = reducer.forward(X_te_dict) if sparse is not None else reducer.forward(X_te_scaled)

    return X_tr_reduced, X_te_reduced


def preprocess(reducer, X_tr, X_te, y_tr, sparse=None):
    """Preprocessing pipeline.
    
    The sparse argument only applies for the AE cv.
    """
    ## Scale and reduce with reducer (if reducer)
    X_tr_reduced, X_te_reduced = scale_and_reduce(
        reducer = reducer,
        X_tr = X_tr,
        X_te = X_te,
        sparse = sparse
    )
    ## Random undersampling and oversampling
    ## NOTE: This should be flipped, i.e. first oversampling, then oversampling.
    undersampler = RandomUnderSampler()
    X_tr_reduced, y_tr = undersampler.fit_resample(X_tr_reduced, y_tr)
    smote = SMOTE()
    X_tr_reduced, y_tr = smote.fit_resample(X_tr_reduced, y_tr)
    
    return X_tr_reduced, X_te_reduced, y_tr


def compute_score(reducer, X, y, cv=StratifiedKFold(n_splits=5), sparse = None):
    """Compute cross-validation score."""
    f1s = []
    for fold, (tr, te) in tqdm(enumerate(cv.split(X=X, y=y))):
        X_tr, y_tr = X.iloc[tr, :], y.iloc[tr]
        X_te, y_te = X.iloc[te, :], y.iloc[te]
        # Standardize, reduce, and oversample
        X_tr_reduced, X_te_reduced, y_tr = preprocess(
            reducer = reducer,
            X_tr = X_tr,
            X_te = X_te,
            y_tr = y_tr,
            sparse = sparse
        )
        # Predict
        clf = LogisticRegression(max_iter = 1000)
        clf.fit(X_tr_reduced, y_tr)
        y_pred = clf.predict(X_te_reduced)
        # Report performance
        f1 = f1_score(
            y_pred=y_pred,
            y_true=y_te,
            average='macro'
        )
        f1s.append(f1)
    
    return np.mean(f1s)