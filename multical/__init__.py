import importlib
import sys
import pkgutil

def import_submodules(package_name):
    """ Import all submodules of a module, recursively

    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]
    """
    package = sys.modules[package_name]
    return {
        name: f(name)
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.')
    }

__all__ = import_submodules(__name__).keys()                        
