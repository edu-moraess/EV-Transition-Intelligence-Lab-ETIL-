def __getattr__(name):
    if name == "Charts":
        from .charts import Charts
        return Charts
    if name == "WorldMap":
        from .world_map import WorldMap
        return WorldMap
    raise AttributeError(f"module 'visualization' has no attribute {name!r}")
