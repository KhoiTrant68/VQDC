import importlib

def instantiate_from_config(config):
    """
    Instantiates an object from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing 'target' (required)
                       and 'params' (optional) keys.
            - 'target' (str): Fully qualified class name (e.g., 'module.class').
            - 'params' (dict, optional): Dictionary of keyword arguments
                                         for the class constructor.

    Returns:
        object: Instantiated object.

    Raises:
        KeyError: If 'target' key is missing in the config dictionary.
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate.")

    # Dynamically import the class from the specified module path
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)

    # Get parameters for class instantiation, defaulting to an empty dict
    params = config.get("params", {})

    # Instantiate the class with provided parameters
    return class_(**params)


def get_obj_from_str(string, reload=False):
    """
    This function is not used in the provided code and can be removed.
    """
    # This function is not used in the provided code, so it's safe to remove.
    raise NotImplementedError("This function is not used and can be removed.") 