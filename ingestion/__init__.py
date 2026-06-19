# ingestion/__init__.py
def __getattr__(name):
    if name in ("IEALoader", "DataLoader"):
        from .loader import IEALoader
        return IEALoader
    if name == "DatasetEngine":
        from .dataset_engine import DatasetEngine
        return DatasetEngine
    if name == "Aggregator":
        from .aggregator import Aggregator
        return Aggregator
    raise AttributeError(f"module 'ingestion' has no attribute {name!r}")
