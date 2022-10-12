# Created by Chen Henry Wu
import os
import configparser
import json


class Args(object):
    def __init__(self, contain=None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name[-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if not (value is None) or (name[:2] == "__" and name[-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        # Place elements in the dictionary order for replicability.
        return sorted(list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__)).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)


def parse_string(string):
    # Integer?
    try:
        return int(string)
    except ValueError:
        pass
    # Float?
    try:
        return float(string)
    except ValueError:
        pass
    # Bool?
    if string in ["True", "true"]:
        return True
    elif string in ["False", "false"]:
        return False
    elif string in ["none", "None"]:
        return None
    # Try JSON.
    try:
        return json.loads(string)
    except json.decoder.JSONDecodeError:
        pass
    # Do not include the \' or \" in the returned string
    return string.strip("\"'")


def get_config(cfg_name):
    args = Args()
    parser = configparser.ConfigParser()
    parser.read(os.path.join('config', cfg_name))
    for section in parser.sections():
        setattr(args, section, Args())
        for key, value in parser.items(section):
            setattr(
                getattr(args, section),
                key,
                parse_string(value),
            )

    return args
