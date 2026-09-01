"""Importing this package registers all built-in adapters under their names."""
from prefscope.data import datasets as _datasets  # noqa: F401
from prefscope.adapters import dataset_openjury  # noqa: F401
from prefscope.interpret import strategy as _interpreter_strategies  # noqa: F401
from prefscope.pipeline import cluster as _clusterers  # noqa: F401
from prefscope.sae import model as _sae_models  # noqa: F401  (registers "sae" components)
