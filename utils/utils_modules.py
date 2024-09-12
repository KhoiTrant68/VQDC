import importlib


def instantiate_from_config(config):
    """Instantiates an object from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary with 'target' and optional 'params' keys.
            - target (str): Fully qualified class name (e.g., 'module.submodule.ClassName').
            - params (dict, optional): Keyword arguments for the class constructor.

    Returns:
        object: Instantiated object.

    Raises:
        KeyError: If 'target' key is missing in the configuration.
    """
    try:
        target_class_name = config["target"]
    except KeyError as e:
        raise KeyError(f"Expected key 'target' to instantiate: {e}")

    module_name, class_name = target_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    target_class = getattr(module, class_name)

    # Pass parameters if provided, otherwise use an empty dictionary
    return target_class(**config.get("params", {}))
