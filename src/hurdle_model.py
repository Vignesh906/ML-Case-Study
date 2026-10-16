"""
Two-Part Hurdle Regression Model for Zero-Inflated Transportation Delays
========================================================================
Specifically designed for commercial aviation and heavy right-skewed transit delay distributions:
Part 1: Calibrated classification predicting P(delay > threshold | X)
Part 2: Non-negative continuous regression predicting E[delay | delay > threshold, X]
Combined: Expected delay = P(delay > threshold | X) * E[delay | delay > threshold, X]

This principled econometric formulation prevents conditional mean inflation caused by
extreme right-tail outliers while remaining responsive to operational risk factors.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

class TwoPartHurdleRegressor(BaseEstimator, RegressorMixin):
    """
    Two-Part Hurdle Regression Model.
    - Part 1: Binary classifier for probability of crossing the on-time threshold (delay > threshold min)
    - Part 2: Continuous regressor for conditional magnitude when delayed
    """
    def __init__(self, threshold=5.0, min_samples_leaf=15, max_iter=100, random_state=42):
        self.threshold = threshold
        self.min_samples_leaf = min_samples_leaf
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.reg = HistGradientBoostingRegressor(
            loss='squared_error',
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

    def fit(self, X, y):
        # Part 1: Hurdle probability
        is_delayed = (y > self.threshold).astype(int)
        self.clf.fit(X, is_delayed)
        
        # Part 2: Magnitude for positive delayed instances
        delayed_mask = (y > self.threshold)
        if np.sum(delayed_mask) >= 20:
            self.reg.fit(X[delayed_mask], y[delayed_mask])
        else:
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        p_delayed = self.clf.predict_proba(X)[:, 1]
        mag_delayed = np.maximum(self.threshold, self.reg.predict(X))
        expected_delay = p_delayed * mag_delayed
        return np.maximum(0.0, expected_delay)

class CoherentTwoPartModel(TwoPartHurdleRegressor):
    """
    Coherent Two-Part Delay Model per mode:
    Part 1: Calibrated classifier predicting P(delay > 0 | X).
    Part 2: Conditional magnitude model predicting E[delay | delay > 0, X].
    Expected delay = P(delay > 0) * E[delay | delay > 0].
    Enforces Markov inequality: E[delay] >= P(delay > 20) * 20.
    """
    def __init__(self, mode="BUS", random_state=42):
        self.mode = mode
        self.random_state = random_state
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        self.p_delayed_clf = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=25, random_state=random_state
        )
        self.cond_mag_reg = HistGradientBoostingRegressor(
            loss='squared_error', max_iter=100, min_samples_leaf=25, random_state=random_state
        )

    def fit(self, X_train, y_train, p_sev_train=None):
        is_delayed = (y_train > 0).astype(int)
        self.p_delayed_clf.fit(X_train, is_delayed)
        
        pos_mask = (y_train > 0)
        if np.sum(pos_mask) >= 50:
            self.cond_mag_reg.fit(X_train[pos_mask], y_train[pos_mask])
        else:
            self.cond_mag_reg.fit(X_train, y_train)
        return self

    def predict(self, X, p_sev=None, p_mod=None):
        p_gt0 = self.p_delayed_clf.predict_proba(X)[:, 1]
        raw_mag = np.maximum(1.0, self.cond_mag_reg.predict(X))
        expected_delay = p_gt0 * raw_mag
        
        # Enforce consistency: E[delay] >= 5*P(Mod) + 20*P(Sev)
        if p_mod is not None and p_sev is not None:
            min_floor = 5.0 * p_mod + 20.0 * p_sev
            expected_delay = np.maximum(expected_delay, min_floor)
        elif p_sev is not None:
            min_floor = p_sev * 20.0
            expected_delay = np.maximum(expected_delay, min_floor)
        return np.maximum(0.0, expected_delay)

    def predict_hurdle_prob(self, X):
        """Returns P(delay > 0 | X)."""
        return self.p_delayed_clf.predict_proba(X)[:, 1]

    def predict_conditional(self, X):
        """Returns E[delay | delay > 0, X]."""
        return np.maximum(1.0, self.cond_mag_reg.predict(X))


class CoherentCalibratedClassifier(BaseEstimator):
    """
    Multimodal Calibrated Delay Classifier.
    Builds P(D>0), P(D>5), P(D>20) from calibrated binary models
    trained WITHOUT balanced class weights.
    Enforces P(D>0) >= P(D>5) >= P(D>20) >= 0 by construction.
    Class probabilities are strictly the differences:
      P(On Time) = 1.0 - P(D>5)
      P(Moderate) = P(D>5) - P(D>20)
      P(Severe) = P(D>20)
    """
    def __init__(self, mode="BUS", random_state=42):
        self.mode = mode
        self.random_state = random_state
        self.classes_ = np.array([0, 1, 2])
        self.base_gt0 = None
        self.cal_gt0 = None
        self.base_gt5 = None
        self.cal_gt5 = None
        self.base_gt20 = None
        self.cal_gt20 = None

    def fit(self, X_tr, d_tr, X_vl, d_vl):
        y0_tr = (d_tr > 0).astype(int)
        y5_tr = (d_tr > 5).astype(int)
        y20_tr = (d_tr > 20).astype(int)

        y0_vl = (d_vl > 0).astype(int)
        y5_vl = (d_vl > 5).astype(int)
        y20_vl = (d_vl > 20).astype(int)

        from sklearn.ensemble import HistGradientBoostingClassifier

        # 1. Binary Model for P(D > 0) - unweighted
        self.base_gt0 = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=25, random_state=self.random_state
        ).fit(X_tr, y0_tr)
        s0_vl = self.base_gt0.predict_proba(X_vl)[:, 1:2]
        self.cal_gt0 = LogisticRegression(C=1.0, random_state=self.random_state).fit(s0_vl, y0_vl)

        # 2. Binary Model for P(D > 5) - unweighted
        self.base_gt5 = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=25, random_state=self.random_state
        ).fit(X_tr, y5_tr)
        s5_vl = self.base_gt5.predict_proba(X_vl)[:, 1:2]
        self.cal_gt5 = LogisticRegression(C=1.0, random_state=self.random_state).fit(s5_vl, y5_vl)

        # 3. Binary Model for P(D > 20) - unweighted
        self.base_gt20 = HistGradientBoostingClassifier(
            max_iter=100, min_samples_leaf=25, random_state=self.random_state
        ).fit(X_tr, y20_tr)
        s20_vl = self.base_gt20.predict_proba(X_vl)[:, 1:2]
        self.cal_gt20 = LogisticRegression(C=1.0, random_state=self.random_state).fit(s20_vl, y20_vl)

        return self

    def predict_cumulative_probs(self, X):
        s0 = self.base_gt0.predict_proba(X)[:, 1:2]
        p0_raw = self.cal_gt0.predict_proba(s0)[:, 1]

        s5 = self.base_gt5.predict_proba(X)[:, 1:2]
        p5_raw = self.cal_gt5.predict_proba(s5)[:, 1]

        s20 = self.base_gt20.predict_proba(X)[:, 1:2]
        p20_raw = self.cal_gt20.predict_proba(s20)[:, 1]

        # Enforce cumulative monotonicity by construction:
        # 1.0 >= P(>0) >= P(>5) >= P(>20) >= 0.0
        p20 = np.clip(p20_raw, 0.0, 1.0)
        p5 = np.clip(np.maximum(p5_raw, p20), 0.0, 1.0)
        p0 = np.clip(np.maximum(p0_raw, p5), 0.0, 1.0)
        return p0, p5, p20

    def predict_p_gt0(self, X):
        p0, _, _ = self.predict_cumulative_probs(X)
        return p0

    def predict_p_gt5(self, X):
        _, p5, _ = self.predict_cumulative_probs(X)
        return p5

    def predict_p_gt20(self, X):
        _, _, p20 = self.predict_cumulative_probs(X)
        return p20

    def predict_proba(self, X):
        p0, p5, p20 = self.predict_cumulative_probs(X)
        p_ontime = np.clip(1.0 - p5, 0.0, 1.0)
        p_mod = np.clip(p5 - p20, 0.0, 1.0)
        p_sev = np.clip(p20, 0.0, 1.0)

        tot = p_ontime + p_mod + p_sev
        tot = np.where(tot > 0, tot, 1.0)
        return np.column_stack([p_ontime / tot, p_mod / tot, p_sev / tot])

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)



