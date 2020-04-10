
import site
from os import path
import json

__all__ = ['get_config_file', 'get_config', 'add_to_config']

try:
    user_base = site.getuserbase()
except AttributeError:
    user_base = path.expanduser('~')

config_file = 'iCCF.conf'

def get_config_file():
    return path.join(user_base, config_file)


def create_config_file():
    conf = get_config_file()
    if not path.exists(conf):
        with open(conf, 'w') as f:
            f.write('{}')


def get_config():
    create_config_file()
    with open(get_config_file()) as f:
        config = json.load(f)
    return config


def add_to_config(**items):
    """ 
    Add key:value pairs to iCCF configuration. They will be stored in the file
    `iCCf.config.get_config_file()`. This will replace any previous values.

    Arguments
    ---------
    items
        Keyword arguments with key:value pairs to add to the configuration.
    
    Example:
        add_to_config(dic)
    """
    config = get_config()
    config.update(items)
    with open(get_config_file(), 'w') as f:
        json.dump(config, f, indent=2)
