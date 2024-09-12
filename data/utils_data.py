class KeyNotFoundError(Exception):
    """Raised if a key is not found in a nested list or dictionary."""

    def __init__(self, cause, keys=None, visited=None):
        """
        Initialize the exception.

        Parameters
        ----------
        cause : Exception
            The underlying exception that caused this error.
        keys : list, optional
            The list of keys that were being searched for.
        visited : list, optional
            The list of keys that were successfully visited before the error
            occurred.
        """

        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Retrieve a value from a nested list or dictionary using a key string.

    The key string describes the path to the desired value using a delimiter.
    For example, the key string 'a/b/c' would retrieve the value at
    list_or_dict['a']['b']['c'].

    If the 'expand' flag is set to True, callable nodes in the path will be
    called. This allows for dynamically generating values in the nested
    structure.

    Parameters
    ----------
    list_or_dict : list or dict
        The nested list or dictionary to retrieve the value from.
    key : str
        The key string describing the path to the desired value.
    splitval : str, optional
        The delimiter used in the key string. Defaults to '/'.
    default : any, optional
        The default value to return if the key is not found. If not specified,
        a KeyNotFoundError is raised.
    expand : bool, optional
        Whether to expand callable nodes in the path. Defaults to True.
    pass_success : bool, optional
        Whether to return a tuple of (value, success) instead of just the value.
        Defaults to False.

    Returns
    -------
    any or tuple
        If 'pass_success' is False, the value at the specified key or the
        default value if the key is not found. If 'pass_success' is True, a
        tuple of (value, success), where 'success' is a boolean indicating
        whether the key was found.

    Raises
    ------
    KeyNotFoundError
        If the key is not found and no default value is specified.
    """

    keys = key.split(splitval)
    success = True

    try:
        visited = []
        parent = None
        last_key = None

        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                if parent is not None:
                    parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]

        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            if parent is not None:
                parent[last_key] = list_or_dict

    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success
