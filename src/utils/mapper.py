""" Implement the ConfigMapper class.
    See class docs for usage."""


class ConfigMapper:
    """Class for creating ConfigMapper objects.

    This class can be used to get and store custom objects from the libraries.
    For each class or object instantiated in any modules,
    the ConfigMapper object can be used either with the functions,
    or as a decorator to store the mapping in the function.

    Examples:

        configmapper = ConfigMapper()

        from torch.nn.optim import Adam
        configmapper.map("optimizers","adam")(Adam)
        adam = configmapper.get("optimizers","adam")(...)
        # Gives the Adam class with corresponding args

        @configmapper.map("datasets","squad")
        class Squad:
            ...

        # This store the `Squad` class to configmapper
        # Can be retrieved and used as

        squad = Squad(...)

    Note: This class has only datasets and schedulers now. The rest can be added as required.

    Attributes:



    Methods
    -------

    """

    dicts = {"datasets": {}, "schedulers": {},}

    @classmethod
    def map(cls, key, name):
        """
        Map a particular name to an object, in the specified key

        Args:
            name (str): The name of the object which will be used.
            key (str): The key of the mapper to be used.
        """

        def wrap(obj):
            if key in cls.dicts.keys():
                cls.dicts[key][name] = obj
            else:
                cls.dicts[key] = {}
                cls.dicts[key][name] = obj
            return obj

        return wrap

    @classmethod
    def get(cls, key, name):
        """Gets a particular object based on key and name.

        Args:
            key (str): [description]
            name (str): [description]

        Raises:
            NotImplementedError: If the key or name is not defined.

        Returns:
            object: The object stored in that key,name pair.
        """
        try:
            return cls.dicts[key][name]
        except KeyError as error:
            if key in cls.dicts:
                raise NotImplementedError(
                    "Key:{name} Undefined in Key:{key}".format(name=name, key=key)
                )
            else:
                raise NotImplementedError("Key:{key} Undefined".format(key=key))


configmapper = ConfigMapper()
