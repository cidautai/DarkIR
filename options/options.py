import os
import yaml
from collections import OrderedDict
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

#-----------------------
Loader, Dumper = OrderedYaml()

def parse(opt_path):
    '''
    Creates a dictionary from the config yaml file.
    '''
    if not os.path.isfile(opt_path): raise ValueError('The config file does not exist!')
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt



if __name__ == '__main__':
    
    path_yaml = './train/NBDN.yml'
    with open(path_yaml, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    opt = parse(path_yaml)
    # print(opt)
    print(type(opt['network']['width']))
    # print(opt['gpu'])