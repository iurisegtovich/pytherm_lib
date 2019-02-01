#will stuff imported here autoreload when using the %autoreload magic?

'''
usage:

1. pip install
...

2. standard import
from example_pkg import rimport_magic

3. etc...
...

'''

#
from example_pkg import version_manifest
phys_file_path_str, modification_time = version_manifest.version_manifest(__file__)
#

#USAGE #%rimport /path/to/pathfile.path module.function

#ARG1 = "../lib_paths/python_utils.path"
#ARG2 = "python_utils.num_utils"

#THE MAGIC SHOULD PARSE, THEN:
#1.
#import sys
#with open( ARG1, "r" ) as f:
#    pkg_path = f.readline()[:-1]
#    sys.path.insert( 1, pkg_path )
#
#2.
#import importlib
#module_name_obj = importlib.import_module( ARG2 ) #hope %autoreloads handles the autoreloading
#'''module_name :: string with complete reference to the module, e.g. "myPackage.aModule.targetSubmodule";
#  to be checked against sys.modules and fed into importlib.reload or importlib.import_module.'''
#
#3. return module_name_obj

#BEGIN>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic, line_cell_magic)
from IPython import get_ipython

@magics_class
class rimport(Magics):
    
    @line_magic
    def rimport(self, line):
        "line magic to rimport"
#
#
#
#
#
        return module_name_obj

# register to make the module have effect in the notebook
ip = get_ipython()
ip.register_magics(rimport)

#END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

