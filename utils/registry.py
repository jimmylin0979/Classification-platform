import torch.nn as nn

# References:
# 1. https://blog.csdn.net/weixin_44347020/article/details/124883366
# 2. https://blog.csdn.net/weiman1/article/details/125610831


class Registry:
    """ """

    def __init__(self, name):
        """
        Args:
            name (str): the name of the registry
        """
        self._name = name
        self._registry = {}

    #### ************* BASE METHODS ************ ####

    def _regsiter(self, name):
        pass

    def register(self, cls, suffix=None):
        """
        Register the given class under the registry name.
        Can be used as either a decorator or not.

        Args:
            cls (type): the class to be registered
            suffix (str): a suffix to add to the class name for naming clarity.
                            This parameter can be used to override the suffix.
        """
        
        # if not issubclass(cls, type):
        #     raise ValueError("{} is not a class!".format(cls))
        if not (suffix is None or isinstance(suffix, str)):
            raise ValueError("Suffix must be a str.")

        cls_name = self._name + suffix if suffix is not None else cls.__name__
        if cls_name in self._registry:
            raise ValueError(
                "{} is already registered in {}".format(cls_name, self._name)
            )

        self._registry[cls_name] = cls

    def get(self, name):
        """
        Args:
            name (str):
        """
        if name not in self._registry:
            raise ValueError(
                "{} not registered in {}".format(name, self.__class__.__name__)
            )
        return self._registry[name]

    #### ************* INSTANCE METHODS ************ ####

    def __contains__(self, name):
        return name in self._registry

    def __getitem__(self, name):
        return self.get(name)

    def __call__(self, name):
        return self.get(name)

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})"
        return format_str.format(self._name, list(self._registry.keys()))

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry.items())


registry = Registry()
