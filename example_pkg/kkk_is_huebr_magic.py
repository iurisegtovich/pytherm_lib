#can i register an ipython magic that reloads a module as the reinitlab does using %reload module ?

'''
usage:

1. pip install
...

2. standard import
from example_pkg import kkk_is_huebr_magic

3. etc...
...

'''

#
from example_pkg import version_manifest
phys_file_path_str, modification_time = version_manifest.version_manifest(__file__)
#

#BEGIN>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from IPython.core.magic import (Magics, magics_class, line_magic, cell_magic, line_cell_magic)
from IPython import get_ipython

@magics_class
class kkkishuebr(Magics):
    
    @line_magic
    def kkkishuebr(self, line):
        "line magic to popullate namespace with kkk=huebr"
        self.shell.user_ns.update({'kkk':'huebr'})
        return

# register to make the module have effect in the notebook
ip = get_ipython()
ip.register_magics(kkkishuebr)

#END<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#references:
# https://ipython.org/ipython-doc/3/config/custommagics.html
# github/pylab
