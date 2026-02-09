# from .reliev3r import Reliev3R

# __all__ = [Reliev3R]

# benchmark/models/reliev3r/model/__init__.py
# Keep package init lightweight to avoid circular imports.

__all__ = ["Reliev3R"]

def __getattr__(name):
    if name == "Reliev3R":
        from .reliev3r import Reliev3R
        return Reliev3R
    raise AttributeError(name)
