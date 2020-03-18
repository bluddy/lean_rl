import os, sys, platform
from os.path import abspath
from os.path import join as pjoin
import signal
import atexit

from xml.dom import minidom
import textwrap
import re
from optparse import OptionParser

# For shared memory
import posix_ipc
import mmap
from ctypes import c_uint8, c_int, POINTER
import numpy as np

import scipy
import scipy.misc

cur_sys = platform.system().lower()
path_sep= ':' if ('linux' in cur_sys or 'darwin' in cur_sys) else ';'
cur_dir= os.path.dirname(abspath(__file__))

isi_path = abspath(pjoin(cur_dir, '..', 'sim', 'ISIH3DModuleBase', 'python'))
os.environ['PYTHONPATH'] = (isi_path + path_sep +
  (os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '') )
sys.path.append(isi_path)

import ISISim.init_env

from SimulationBase import SimulatorConnection
from SimulationBase import NetworkHelper
from SimulationBase import Settings


settings = Settings.Settings( "NetworkClient", local_settings=True)

parser = OptionParser()
parser.add_option( "-p", "--port", dest="port", default=60001, type="int",
                  help="The port to connect to" )
parser.add_option( "-a", "--host", dest="host", default="localhost",
                  help="The host to connect to" )
parser.add_option( "", "--eventhost", dest="eventhost", default="localhost",
                  help="The host to which the server should send event notifications, this should be your host where this client is running" )
parser.add_option( "", "--eventport", dest="eventport", default=60005,
                  help="The port to use to listen for upstream events" )
( options, args ) = parser.parse_args()

param_mode = False

parameter_name_help = "Enter the name of the parameter to add, or just hit return to send the action now:"
parameter_value_help = "Enter the parameter value (which may be multiple lines), then Ctrl-C to end:"

through_daemon = True
if( options.port != 60000 ):
  through_daemon = False

print "port=", options.port, " host=", options.host
print "eventport=", options.eventport, " eventhost=", options.eventhost

# The simulator connection to use to send/recieve data
sim_connection = SimulatorConnection.SimulatorConnection( options.host, options.port, via_daemon=through_daemon )

# Create cleanup function, set as SIGINT signal handler
def cleanup(a, b):
  global sim_connection
  sim_connection.__del__()
  sim_connection = None
  print "Ending program"
  import time
  time.sleep(500)
  exit(1)

signal.signal(signal.SIGINT, cleanup)
#atexit.register(cleanup, 0,0)

def sanitize_whitespace( text ):
  def repl( match ): # avoid "unmatched group" error
    g = match.groups()
    # basically "\1\2\3"
    return ( g[0] or '' ) + ( g[1] or '' ) + ( g[2] or '' )
  return sanitize_whitespace.pattern.sub( repl, text )
sanitize_whitespace.pattern = re.compile( r"""(?mxu)
    # spaces before a line - capture the linebreak in 1
    (^)[\ \t]+   |
    # spaces after a line - capture the linebreak in 2
    [\ \t]+($)   |
    # multiple spaces within a line - capture the replacement space in 3
    ([\ \t])[\ \t]+
""" )

def listActions():

  # Get a full list of all the API actions available
  available_simulation_actions = []
  result = callAction( NetworkHelper.Action( "getAPI" ) )
  if result.isSuccess():
    api_dom = minidom.parseString( result.getContentText() )
    available_simulation_actions = api_dom.getElementsByTagName( "Action" )

  # If we're connected to a daemon, also list Daemon API
  available_actions = []
  result = callAction( NetworkHelper.Action( "getAPI", "Daemon" ) )
  if result.isSuccess():
    api_dom = minidom.parseString( result.getContentText() )
    available_actions = api_dom.getElementsByTagName( "Action" )

  print "The following daemon actions are available: "
  print ""
  for action in available_actions:
    print action.getAttribute( "name" )
    description = NetworkHelper.getText( action )
    if description != "":
      print "----------------"
      printWrap( description )
      print "----------------"
    print ""

  if available_simulation_actions:
    print "The following simulation actions are available: "
    print ""
    for action in available_simulation_actions:
      print action.getAttribute( "name" )
      description = NetworkHelper.getText( action )
      if description != "":
        print "----------------"
        printWrap( description )
        print "----------------"
      print ""

  print "Type an action name, then any parameters (in Python syntax)"
  print "separated by semi-colons."
  print ""
  print "e.g. 'getAPIName', 'enableGraphics False',"
  print "     'setShapeColor RGB(1,0,0)'"
  print "     'displayMessageColor \"Hello world!\"; RGB(1,0,0)'"
  print ""
  print "To add <Parameter/> elements to the action type !param first."
  print ""

def toggleParamMode():
  global param_mode
  param_mode = not param_mode
  if param_mode:
    print "In parameter mode (type '!param' again to exit)"
  else:
    print "Left parameter mode (type '!param' again to enter)"

def listMetrics():

  result = callAction( NetworkHelper.Action( "getMetrics" ) )
  if result.isSuccess():
    dom = minidom.parseString( result.getContentText() )
    for m in dom.getElementsByTagName( "Metric" ):
      desc = m.getElementsByTagName( "Description" )
      val = m.getElementsByTagName( "Value" )
      longDesc = m.getElementsByTagName( "LongDescription" )
      print "----------------"
      if len( desc )>0:
        print NetworkHelper.getText( desc[0] )
      if len( val )>0:
        print NetworkHelper.getText( val[0] )
      if len( longDesc )>0:
        print NetworkHelper.getText( longDesc[0] )
      print "----------------"
      print ""
  else:
    print "Could not get metrics."
    print ""
    print formatActionResult( result )

def printWrap( text ):
  print textwrap.fill( sanitize_whitespace( text ) )

def callAction( action ):
  r = sim_connection.call( action )
  return r

def callbackDisplayMessage( condition_id, message ):
  print ""
  print str( message )
  print "The id of the condition which triggered the callback is " + str( condition_id )
  print ""

def formatActionResult( action_result ):
  if action_result.isSuccess():
    s = "The action completed successfully.\n"
    if action_result.getContentText() != "":
      s+= "\nData:\n" + action_result.getContentText() + "\n"
  else:
    s = "There was an error executing the action.\n"
    if action_result.getDescription() != "":
      s+= "\nDescription: " + action_result.getDescription() + "\n"
  return s

def displayMessage( message ):
  print ""
  print str( message )
  print ""

sim_time_trigger_4 = None
sim_time_trigger_8 = None
sim_time_trigger_12 = None
sim_time_trigger_16 = None

SHARED_PATH = "/H3D"
VAR_NUM = 3  # Variables before array
VAR_SIZE = VAR_NUM * 4  # size of variables
shared_mem = None
mmap_shared = None
shared_vars = None
shared_array = None
ready_flag = None
ready_flag_old = None
img_count = 1

def testDataEvent():
  ''' Create a callback to respond to the Event from the sim
      @return new callback_id
  '''

  def callback(event):
    global shared_mem, mmap_shared, shared_array, shared_vars, img_count, ready_flag, ready_flag_old

    if shared_mem is None:
      print "Initializing shared memory"
      shared_mem = posix_ipc.SharedMemory(SHARED_PATH)
      mmap_shared = mmap.mmap(shared_mem.fd, 0)
      shared_vars = (c_int * VAR_NUM).from_buffer(mmap_shared)
      s_type = c_uint8 * (mmap_shared.size() - VAR_SIZE)
      shared_array = s_type.from_buffer(mmap_shared, VAR_SIZE)
      ready_flag = shared_vars[0]
      ready_flag_old = ready_flag
    else:
      # Check if we got new image data
      ready_flag = shared_vars[0]
      print "ready flag :", ready_flag, ", ready_flag_old :", ready_flag_old

      # Check if stuff has been written
      if ready_flag != ready_flag_old:
        width = shared_vars[1]
        height = shared_vars[2]
        arr = np.ctypeslib.as_array(shared_array)
        arr = np.reshape(arr, (height, width, 3))
        # Save to png
        print "Received image ", img_count
        scipy.misc.imsave('./img{}.png'.format(img_count), arr)
        img_count += 1

        ready_flag_old = ready_flag

    print "Received Event: ", event.getContentText()

  callback_id = sim_connection.registerSimpleEvent(callback)
  result = sim_connection.startCallbackListener(options.eventhost, int(options.eventport))
  if not result.isSuccess():
    print "ERROR: SimulatorConnection::startCallbackListener could NOT be performed successfully."
    print result.getDescription()
    print result.getContentText()
  return callback_id


# Get the simulation version string
result = callAction( NetworkHelper.Action( "getAPIName" ) )
if result.isSuccess():
  apiName = result.getContentText()
else:
  print "Could not connect to network API."
  print ""
  print formatActionResult( result )
  sys.exit()

# Create a callback and tell the simulation about it (event)
callback_id = testDataEvent()
print "Callback id is ", callback_id
action = NetworkHelper.Action("setEventId", "", [str(callback_id)])
#print "_parameters = ", action._parameters
print formatActionResult(callAction(action))

print ""
print "Connected to " + apiName
print ""
print "Type '!help' to list available API actions."
print "Type '!quit' to exit."
print ""

input = raw_input( ">" )
while input != "!quit":
  if input == "!help":
    listActions()
  elif input == "!param":
    toggleParamMode()
  elif input == "!metrics":
    listMetrics()
  else:
    input = input.split( None, 1 )
    name = None
    parameters = ''

    if len( input )>0:
      name_and_path = input[0]
      namespaces = name_and_path.split( "." )
      name = namespaces[-1]
      path = ".".join( namespaces[:-1] )

      if len( input )>1:
        parameters = input[1].split( ";" )

      print "name =", name
      print "path =", path
      print "parameter =", parameters
      action = NetworkHelper.Action( name, path, parameters )
      print action._parameters

      if param_mode:
        params = []
        printWrap( parameter_name_help )
        param_name = raw_input( ">>" )
        while param_name != "":
          paramValue = ""
          printWrap( parameter_value_help )
          try:
            while True:
              paramValue+= raw_input( ">>" )
          except KeyboardInterrupt:
            print ""
          action.addParameter( paramValue )
          printWrap( parameter_name_help )
          param_name = raw_input( ">>" )

      print formatActionResult( callAction( action ) )

  input = raw_input( ">" )
