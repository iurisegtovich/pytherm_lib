#module other_module

#
from example_pkg import version_manifest
phys_file_path_str, modification_time = version_manifest.version_manifest(__file__)
#

def splash():
  print('nothing happened')
