import os
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


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


def mark_prepared(root):
    """Create a '.ready' file in the given directory to mark it as prepared.

    Parameters
    ----------
    root : str
        The path to the directory to mark as prepared.
    """
    (Path(root) / ".ready").touch()


def is_prepared(root):
    """Check if a directory is marked as prepared.

    Parameters
    ----------
    root : str
        The path to the directory to check.

    Returns
    -------
    bool
        True if the directory is marked as prepared, False otherwise.
    """
    return (Path(root) / ".ready").exists()


def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    """Return a list of synsets corresponding to the given indices.

    The mapping from indices to synsets is read from a YAML file.

    Parameters
    ----------
    indices : list
        A list of indices for which to retrieve the synsets.
    path_to_yaml : str, optional
        The path to the YAML file containing the index-to-synset mapping.
        Defaults to 'data/imagenet_idx_to_synset.yaml'.

    Returns
    -------
    list
        A list of synsets corresponding to the given indices.
    """

    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.safe_load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print(
        "Using {} different synsets for construction of Restriced Imagenet.".format(
            len(synsets)
        )
    )
    return synsets


def str_to_indices(string):
    """Convert a string of comma-separated index ranges to a list of integers.

    The input string should be in the format '32-123, 256, 280-321'. Each
    element in the string can be either a single index or a range of indices
    separated by a hyphen.

    Parameters
    ----------
    string : str
        A string of comma-separated index ranges.

    Returns
    -------
    list
        A sorted list of integers representing the indices.

    Raises
    ------
    AssertionError
        If the input string ends with a comma or if an element in the string
        is not a valid index or range.
    """
    assert not string.endswith(
        ","
    ), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]) + 1)]
            indices.extend(rang)
    return sorted(indices)


def download(url, local_path, chunk_size=1024):
    """Download a file from a given URL to a local path.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    local_path : str
        The local path to save the downloaded file.
    chunk_size : int, optional
        The size of each chunk to download in bytes. Defaults to 1024.
    """
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(len(data))


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
