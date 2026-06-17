import pytest
from ingestion.loader import DataLoader

def test_loader_instantiation():
    loader = DataLoader()
    assert loader is not None
