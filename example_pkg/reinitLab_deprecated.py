#can i register an ipython magic that reloads a module as the reinitlab does using %reload module ?

'''
usage:

1. pip install
...

2. standard import
from python_utils import m_reinitLab

3. self test
m_reinitLab = m_reinitLab.reinitLab("python_utils.m_reinitLab")

4. have fun

5. magic

'''

#
from example_pkg import version_manifest
phys_file_path_str, modification_time = version_manifest.version_manifest(__file__)
#

def reinitLab(module_name,varsdict=None,echo=False):
    '''module_name :: string with complete reference to the module, e.g. "myPackage.aModule.targetSubmodule";
                      to be checked against sys.modules and fed into importlib.reload or importlib.import_module.
       varsdict :: to be fed with vars() from local scope,
                   will update the varsdict object as if "from myPackage.aModule.targetSubmodule import *" was run in the caller scope,
                   but ignoring variables starting with _
                   returns the filtereddict, which was used to update the varsdict,
                   there is no need to bind it in the caller scope except for debugging purposes;
                   if None, returns the module object as if "import myPackage.aModule.targetSubmodule as _*" was run in the caller scope.
       echo :: if True, tries to print the _manifest variable of the target module.
       
       usage1: >>> m_reinitLab = m_reinitLab.reinitLab("python_utils.m_reinitLab")
       
       usage2: >>> m_reinitLab.reinitLab("python_utils.m_reinitLab",vars())
       
    '''
    import importlib
    import sys
    
    if module_name in sys.modules:
        module_name_obj=importlib.reload(sys.modules[module_name])
    else:
        module_name_obj=importlib.import_module(module_name)
    
    if echo:
        print(module_name_obj._manifest)
    
    if varsdict is not None:
        filtereddict={k: v for k, v in module_name_obj.__dict__.items() if k[0] != "_"}
        varsdict.update(filtereddict) #update as in "from m_HydLab import *"
        return filtereddict
    else:
        return module_name_obj#returns either newly imported module or updated existing
        
        
        
#1 pylab magic https://github.com/ipython/ipython/search?q=pylab&unscoped_q=pylab
#2 python tutor magic (eqe359)

#    @magic_gui_arg
#    def pylab(self, line=''):

# gui, backend, clobbered = self.shell.enable_pylab(args.gui, import_all=import_all)

#    def enable_pylab(self, gui=None, import_all=True, welcome_message=False):
#        # We want to prevent the loading of pylab to pollute the user's
#        # namespace as shown by the %who* magics, so we execute the activation
#        # code in an empty namespace, and we update *both* user_ns and
#        # user_ns_hidden with this information.
#        ns = {}
#        import_pylab(ns, import_all)
#        self.user_ns.update(ns)

# def import_pylab(user_ns, import_all=True):
# exec(s, user_ns)

#https://ipython.org/ipython-doc/3/config/custommagics.html

#        print("Variables in the user namespace:", list(self.shell.user_ns.keys()))
# AQUI PODE ATUALIZAR, NÃO É HACK DE PYTHON, É MAGICA DE IPYTHON
