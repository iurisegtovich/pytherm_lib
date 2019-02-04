#can i register an ipython magic that reloads a module as the reinitlab does using %reload module ?

'''
usage:

1. pip install
...

2. standard import
from example_pkg import reload_magic

3. etc...
...

'''

#
from example_pkg import version_manifest
phys_file_path_str, modification_time = version_manifest.version_manifest(__file__)
#

# ? is this still relevant after learning about:
#
#%load_ext autoreload

#%autoreload 1

#    Reload all modules imported with %aimport every time before executing the Python code typed.

#%aimport foo

#    Import module ‘foo’ and mark it to be autoreloaded for %autoreload 1

#%aimport

#    List modules which are to be automatically imported or not to be imported.
#
