'''
USAGE:

import print_file_line
print_file_line.print_file_line(print_file_line.gibe_frame()())

or

from print_file_line import print_file_line, gibe_frame
print_file_line(gibe_frame()())

'''

#these can be included/sourced to your file,
#they wont work as function

#from inspect import currentframe, getframeinfo
#frameinfo = getframeinfo(currentframe())
#print('Hello from: '+frameinfo.filename+' @ '+str(frameinfo.lineno))

#unless it was a hack like this:

def gibe_frame():
  from inspect import currentframe
  return currentframe

def print_file_line(currentframe_called):
  from inspect import getframeinfo
  frameinfo = getframeinfo(currentframe_called)
  print('Hello from: '+frameinfo.filename+' @ '+str(frameinfo.lineno))


#then call like
# print_file_line(gibe_frame()())
