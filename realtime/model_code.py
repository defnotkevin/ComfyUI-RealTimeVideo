"""Import installed, trusted transformer code without Diffusers' shared cache."""
import hashlib
import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import types


def load_transformer_class(model):
    directory = (Path(model).resolve() / 'transformer')
    for filename in ('causal_model.py', 'attention.py', 'model.py'):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f'Missing installed Krea code: {directory / filename}. Re-run the model downloader.')
    # Give each installation its own package. Python resolves all relative
    # imports directly in that directory, including transitive dependencies.
    name = '_krea_transformer_' + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        package.__package__ = name
        package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[name] = package
    importlib.invalidate_caches()
    return importlib.import_module(name + '.causal_model').CausalWanModel
