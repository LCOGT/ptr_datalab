from abc import ABC, abstractmethod
from pkgutil import walk_packages
import inspect

from datalab.datalab_session import data_operations


def available_operations():
    operations = {}
    for (loader, module_name, _) in walk_packages(data_operations.__path__):
        module = loader.find_module(module_name).load_module()
        members = inspect.getmembers(module, inspect.isclass)
        for member in members:
            if member[0] != 'BaseDataOperation' and issubclass(member[1], BaseDataOperation):
                operations[member[0]] = member[1]

    return operations


def available_operations_tuples():
    names = available_operations().keys()
    return [(name, name) for name in names]


class BaseDataOperation(ABC):

    @staticmethod
    @abstractmethod
    def name():
        """ A unique name for your DataOperation """

    @staticmethod
    @abstractmethod
    def description():
        """ A text description of the DataOperation, to be shown to the user """

    @staticmethod
    @abstractmethod
    def wizard_description():
        """ A json-formatted DSL describing the expected inputs for this DataOperation,
            for the frontend to create custom input widgets for it in a wizard
        """

    @abstractmethod
    def operate(self, input_data):
        """ The method that performs the data operation. The data inputs are passed in in the format described from the wizard_description """
