"""
Bagged-tree next-step predictor for symbolic sequences
------------------------------------------------------
Input : any iterable of hashable symbols (ints, str, …)
Output: predicted next symbol + probability vector
Author: <you>
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from collections import Counter

# ------------------------------------------------------------------
# 1.  Turn the raw sequence into a supervised table
# ------------------------------------------------------------------
def embed_sequence(seq, k=4):
    """
    seq : list/array of symbols
    k   : look-back window length
    returns
        X : (N-k, k)  – each row is the k-gram
        y : (N-k,)    – the symbol that follows that k-gram
    """
    seq = np.asarray(seq)
    if len(seq) <= k:
        raise ValueError("Sequence must be longer than look-back window k")
    X = np.lib.stride_tricks.sliding_window_view(seq, k)[:-1]
    y = seq[k:]
    return X, y

# ------------------------------------------------------------------
# 2.  Build a bagged tree ensemble
# ------------------------------------------------------------------
def build_forecaster(X, y, n_trees=500, max_depth=None,
                     random_state=None, oob_score=True):
    """
    Returns a fitted BaggingClassifier that can .predict() and .predict_proba()
    """
    base = DecisionTreeClassifier(max_features=None,
                                  max_depth=max_depth,
                                  random_state=random_state)
    bag = BaggingClassifier(estimator=base,
                            n_estimators=n_trees,
                            max_samples=1.0,
                            bootstrap=True,
                            oob_score=oob_score,
                            n_jobs=-1,
                            random_state=random_state)
    bag.fit(X, y)
    if oob_score:
        print(f"Out-of-bag accuracy : {bag.oob_score_:.3f}")
    return bag

# ------------------------------------------------------------------
# 3.  Convenience wrapper
# ------------------------------------------------------------------
class SequenceForecaster:
    def __init__(self, k=4, n_trees=500, max_depth=None, random_state=42):
        self.k = k
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.alphabet = None

    def fit(self, seq):
        X, y = embed_sequence(seq, self.k)
        self.alphabet = np.unique(y)          # symbols that actually appear
        self.model = build_forecaster(X, y,
                                      n_trees=self.n_trees,
                                      max_depth=self.max_depth,
                                      random_state=self.random_state)
        return self

    def predict(self, context):
        context = np.asarray(context)
        if len(context) != self.k:
            raise ValueError(f"Context must contain exactly k={self.k} symbols, got {len(context)}")
        probs = self.model.predict_proba(context.reshape(1, -1))[0]
        best  = self.model.classes_[np.argmax(probs)]
        return best, dict(zip(self.model.classes_, probs))

# ------------------------------------------------------------------
# 4.  Demo on the user-supplied sequence
# ------------------------------------------------------------------
if __name__ == "__main__":
    raw = [1, 4, 8, 12, 16, 20, 24]   # whatever your real data are
         # extend as needed: 14812162024…
    k   = 3                          # use 3-grams (tune freely)

    fc = SequenceForecaster(k=k, n_trees=500).fit(raw)

    # predict next symbol after the last k symbols
    last_k = raw[-k:]
    next_sym, probs = fc.predict(last_k)
    print("Last k symbols :", last_k)
    print("Predicted next :", next_sym)
    print("Probability dist:", probs)
