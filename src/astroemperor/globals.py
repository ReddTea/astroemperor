# @auto-fold regex /^\s*if/ /^\s*else/ /^\s*def/
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import platform
import multiprocessing

_PLATFORM_SYSTEM = platform.system()  # PROBABLY BEST, should bench
_SYS_PLATFORM = sys.platform
_OS_NAME = os.name
_CORES = multiprocessing.cpu_count()
_OS_ROOT = os.path.dirname(__file__)  # where im now

# TERMINAL WIDTH
try:
    _TERMINAL_WIDTH = os.get_terminal_size().columns
except:
    print('I couldnt grab the terminal size. Trying with pandas...')
    try:
        import pandas
        _TERMINAL_WIDTH = pandas.get_option('display.width')
        print('Terminal size with pandas successful!')
    except:
        print('Failed to grab the terminal size with pandas :(')

# SHELL ENV
try:
    shl = get_ipython().__class__.__name__
    if shl == 'ZMQInteractiveShell':
        _SHELL_ENV = 'jupyter-notebook'
    elif shl == 'TerminalInteractiveShell':
        _SHELL_ENV = 'ipython-terminal'
    elif get_ipython().__class__.__module__ == 'google.colab._shell':
        _SHELL_ENV = 'google-colab'
        
except NameError:
    _SHELL_ENV = 'python-terminal'