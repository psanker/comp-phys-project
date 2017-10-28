#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-

#  _____        _        __  __           _
# |  __ \      | |      |  \/  |         | |
# | |  | | __ _| |_ __ _| \  / | __ _ ___| |_ ___ _ __
# | |  | |/ _` | __/ _` | |\/| |/ _` / __| __/ _ \ '__|
# | |__| | (_| | || (_| | |  | | (_| \__ \ ||  __/ |
# |_____/ \__,_|\__\__,_|_|  |_|\__,_|___/\__\___|_|

# A project module loader for unified data analysis

import signal
import sys
import gc
import getopt
import os

# importing matplotlib so datamaster prints all requested plots at once
import matplotlib.pyplot as plt

cli_thread   = True # If false, the CLI terminates; goes false upon SIGKILL

PROJECTS         = []   # List of all known projects given by subdirectories
current_projects = {}   # Key-value memory of active projects
selected_project = None # Name of current selected project; used for reloads

VERSION      = '1.2.0' # Current version of DataMaster

def fetch_project(name, load):
    '''
    * Internal *

    Attempts to load a project given an input name
    Returns the initialized project module or None if an error is encountered
    '''
    obj = None

    if name not in PROJECTS:
        # Directly return None instead of obj
        print('Invalid project name')
        return None

    if name in current_projects and not load:
        if current_projects[name] is not None:
            print('%s loaded from memory' % (name))
            obj = current_projects[name]

    elif name in current_projects and load:
        try:
            unload_project(name)

            obj = load_project(name)

            if obj is None:
                print('Reload of %s failed; Try loading directly from file?' % (name))

                unload_project(name)
                return None
            else:
                current_projects[name] = obj

        except Exception as err:
            print('Reload of project failed')
            print(str(err))
    else:
        obj = load_project(name)

        if obj is not None:
            current_projects[name] = obj
        else:
            print('Could not load \'%s\'' % (name))

    return obj

def load_project(name):
    obj = None

    try:
        obj = __import__(name)

        if not hasattr(obj, 'project') and not hasattr(obj, 'lab'):
            print('__init__ file not properly configured.')
            obj = None
        else:
            current_projects[name] = obj
    except Exception as err:
        print('Project could not be loaded')
        print(str(err))

        # Resets the object reference in case something weird happened
        obj = None

    return obj

def unload_project(name):
    '''
    * Internal *

    Removes all references of a project from memory
    '''

    # Firstly, if the object has terminate(), call it; useful for freeing memory
    obj = current_projects[name]

    if hasattr(obj.project, 'terminate') and callable(getattr(obj.project, 'terminate')):
        getattr(obj.project, 'terminate')()

    # Now, purge references
    rm = []

    global selected_project

    if str(selected_project) == name:
        selected_project = None

    # Filter submodules that belong to target project, if it exists
    for mod in sys.modules.keys():
        if mod.startswith('%s.' % (name)):
            rm.append(mod)

    for i in rm:
        sys.modules[i] = None
        del sys.modules[i]

    # Redundant checks in case reference is broken
    if name in sys.modules:
        sys.modules[name] = None
        del sys.modules[name]

    if name in current_projects:
        del current_projects[name]

    gc.collect()

def select_project(name, load=False):
    project = fetch_project(name, load)

    if project is not None:
        global selected_project
        selected_project = name
        print('Selected Project: %s\n' % (selected_project))

def plot_var(var):
    if selected_project is not None:
        obj = current_projects[selected_project]

        if str(var) == 'all':

            for e in dir(obj.project):
                if e.startswith('plot_') and callable(getattr(obj.project, str(e))):
                    getattr(obj.project, str(e))()

            return True

        try:
            getattr(obj.project, ('plot_%s' % (var)))()
            return True
        except Exception as err:
            print(str(err))
    else:
        print('No selected project')
        usage()
        return False

def get_var(var):
    if selected_project is not None:
        obj = current_projects[selected_project]

        if str(var) == 'all':

            for e in dir(obj.project):
                if e.startswith('get_') and callable(getattr(obj.project, str(e))):
                    print(getattr(obj.project, str(e))())

            return

        try:
            print(getattr(obj.project, ('get_%s' % (var)))())
        except Exception as err:
            print(str(err))
    else:
        print('No selected project')
        usage()

def run_func(var):
    if selected_project is not None:
        obj = current_projects[selected_project]

        if str(var) == 'all': # Are you sure about that?

            for e in dir(obj.project):
                if e.startswith('run_') and callable(getattr(obj.project, str(e))):
                    getattr(obj.project, str(e))()
            return

        try:
            getattr(obj.project, ('run_%s' % (var)))()
        except Exception as err:
            print(str(err))
    else:
        print('No selected project')
        usage()

def list():
    print('Available projects:')

    for s in PROJECTS:
        if selected_project is not None and s == selected_project:
            print('> * %s' % (s))
        else:
            print('  * %s' % (s))

    if selected_project is not None:
        obj   = current_projects[selected_project]

        gets  = []
        plots = []
        runs  = []

        for e in dir(obj.project):
            if e.startswith('get_') and callable(getattr(obj.project, str(e))):
                gets.append(str(e).replace('get_', ''))
            elif e.startswith('plot_') and callable(getattr(obj.project, str(e))):
                plots.append(str(e).replace('plot_', ''))
            elif e.startswith('run_') and callable(getattr(obj.project, str(e))):
                runs.append(str(e).replace('run_', ''))

        if len(gets) != 0 or len(plots) != 0 or len(runs) != 0:
            print('----------------------------------')
            print('Functions for \'%s\'' % (selected_project))

            if len(gets) != 0:
                print('Gets:')

                for g in gets:
                    print ('  * %s' % (g))

            if len(plots) != 0:
                print('Plots:')

                for p in plots:
                    print ('  * %s' % (p))

            if len(runs) != 0:
                print('Runnables:')

                for r in runs:
                    print ('  * %s' % (r))

def current_version():
    print('DataMaster version %s' % VERSION)

def usage():
    current_version()
    print('Usage: datamaster.py -s <name> [-g, -p] <data name>')
    print('\nCommands:\n  -h, --help: Prints out this help section')
    print('  -l, --list: Lists all the available projects and, if a project is selected, all available gets and plots')
    print('  -s, --select <name>: Selects project to compute data from')
    print('  -r, --reload: Reloads the selected project from file')
    print('  -p, --plot <variable>: Calls a plotting function of the form \"plot_<variable>\"')
    print('  -g, --get <variable>: Prints out a value from function of the form \"get_<variable>\"')
    print('  -x, --run <variable>: Runs a custom function of the form \"run_<variable>\"')
    print('  -e, --exit: Explicit command to exit from DataMaster CLI')

def cli():
    legacy = False

    if sys.version_info < (3, 0):
        legacy = True

    while cli_thread:
        if legacy:
            # Py ≤ 2.7
            args = str(raw_input('> ')).split(' ')
            handle_args(args)
        else:
            # Py 3
            args = str(input('> ')).split(' ')
            handle_args(args)

def exit_handle(sig, frame):
    global cli_thread
    cli_thread = False # Safely halts while loop in thread

    print('\nExiting...')
    sys.exit(0)

def handle_args(args):
    try:
        opts, args = getopt.getopt(args, 'hlvs:rp:g:x:e', ['help', 'list', 'version', 'reload', 'select=', 'plot=', 'get=', 'run=', 'exit'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        return

    plotting = False

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            return

        elif opt in ('-l', '--list'):
            list()
            return

        elif opt in ('-v', '--version'):
            current_version()
            return

        elif opt in ('-s', '--select'):
            select_project(arg)

        elif opt in ('-r', '--reload'):
            if selected_project is not None:
                select_project(selected_project, True)
            else:
                print('No selected project to reload')
                return

        elif opt in ('-p', '--plot'):
            if plot_var(arg) and plotting is False:
                plotting = True

        elif opt in ('-g', '--get'):
            get_var(arg)

        elif opt in ('-x', '--run'):
            run_func(arg)

        elif opt in ('-e', '--exit'):
            print('\nExiting...')
            sys.exit(0)

        else:
            usage()

    if plotting:
        plt.show()
        plt.close('all')

def main(argv):
    # Load in project names
    global PROJECTS
    PROJECTS = [name for name in os.listdir('.') if os.path.isdir(name) and os.path.exists(os.path.join(name, 'project.py'))]

    if len(argv) == 0:
        # register sigkill event and start looping CLI
        signal.signal(signal.SIGINT, exit_handle)
        cli()
    else:
        handle_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
