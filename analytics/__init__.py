def __getattr__(name):
    if name == "CountryAnalyzer":
        from .country_analysis import CountryAnalyzer
        return CountryAnalyzer
    if name == "GlobalRanking":
        from .global_ranking import GlobalRanking
        return GlobalRanking
    if name == "ScenarioEngine":
        from .scenario_engine import ScenarioEngine
        return ScenarioEngine
    raise AttributeError(f"module 'analytics' has no attribute {name!r}")
