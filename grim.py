# grim.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class GreedyRuleInterpretableMachine(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=1000, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Call build_grim() from inside the class
        self._rf_model = self.build_grim()

        self.rule_list_ = None

    def build_grim(self):
        """
        Build internal model (called from external file too).
        """
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self._rf_model.fit(X, y)
        self._generate_rule_list()
        return self

    def predict(self, X):
        return self._rf_model.predict(X)

    def predict_proba(self, X):
        return self._rf_model.predict_proba(X)

    def _generate_rule_list(self):
        rules = []
        for estimator in self._rf_model.estimators_[:5]:
            tree = estimator.tree_
            feature = tree.feature
            rules.append(
                f"GRIM Rule → Tree depth={tree.max_depth}, "
                f"Features used={np.unique(feature[feature >= 0]).size}"
            )
        self.rule_list_ = rules

    def get_rule_list(self):
        if self.rule_list_ is None:
            raise ValueError("Model must be fitted before extracting GRIM rule list.")
        return self.rule_list_

    def score(self, X, y):
        return self._rf_model.score(X, y)
