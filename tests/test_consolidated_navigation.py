import importlib
import inspect


def test_model_lab_workflows_support_embedded_rendering():
    modules = [
        "streamlit_pages.03_Diffusion_Engine",
        "streamlit_pages.04_Regime_Detection",
        "streamlit_pages.05_Forecast_Lab",
        "streamlit_pages.09_Scenarios",
    ]
    for name in modules:
        render = importlib.import_module(name).render
        assert "embedded" in inspect.signature(render).parameters


def test_signal_research_workflows_support_embedded_rendering():
    modules = [
        "streamlit_pages.06_EVQTI_Ranking",
        "streamlit_pages.10_ETII",
        "streamlit_pages.11_Research_Intelligence",
    ]
    for name in modules:
        render = importlib.import_module(name).render
        assert "embedded" in inspect.signature(render).parameters


def test_consolidated_workspace_modules_are_importable():
    assert callable(importlib.import_module("streamlit_pages.20_Model_Lab").render)
    assert callable(importlib.import_module("streamlit_pages.21_Signals_Research").render)
