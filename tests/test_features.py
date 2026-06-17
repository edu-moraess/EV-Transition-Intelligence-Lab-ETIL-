import pandas as pd
from preprocessing.features import FeatureEngineer

def test_feature_engineer():
    df = pd.DataFrame({
        "country": ["A", "A"],
        "year": [2020, 2021],
        "ev_sales": [10, 20],
        "total_sales": [100, 200]
    })
    engineer = FeatureEngineer(df)
    df_out = engineer.engineer()
    assert "penetration" in df_out.columns
