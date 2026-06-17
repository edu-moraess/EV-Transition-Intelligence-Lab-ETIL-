# models/regimes/__init__.py
def __getattr__(name):
    if name == "HMMRegimeDetector":
        from .hmm import HMMRegimeDetector
        return HMMRegimeDetector
    if name == "BayesianChangePoint":
        from .changepoint import BayesianChangePoint
        return BayesianChangePoint
    raise AttributeError(f"module 'models.regimes' has no attribute {name!r}")
