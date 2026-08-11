from pkgutil import walk_packages
import inspect
from importlib import import_module
from django.utils.module_loading import import_string
from datalab.datalab_session import data_operations


def available_operations():
    operations = {}
    base_operation = import_string('datalab.datalab_session.data_operations.data_operation.BaseDataOperation')
    for (loader, module_name, _) in walk_packages(data_operations.__path__):
        module = import_module(f'{data_operations.__name__}.{module_name}')
        members = inspect.getmembers(module, inspect.isclass)
        for member in members:
            # Abstract subclasses are shared implementation, not operations to offer.
            if issubclass(member[1], base_operation) and not inspect.isabstract(member[1]):
                operations[member[1].name()] = member[1]

    return operations


def available_operations_tuples():
    names = available_operations().keys()
    return [(name, name) for name in names]
